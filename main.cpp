#define _USE_MATH_DEFINES
#include <vector>
#include <iostream>

#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

Model *model     = NULL;
float* shadowbuffer = NULL;

const int width = 800;
const int height = 800;
Vec3f light_dir(1, 0, 1);
Vec3f       eye(0, 0, 4);
Vec3f    center(0, 0, 0);
Vec3f        up(0, 1, 0);
using namespace std;
struct GouraudShader : public IShader {
    Vec3f varying_intensity; // written by vertex shader, read by fragment shader
         // written by vertex shader, read by fragment shader
    mat<3, 3, float> varing_verts;

    mat<2, 3, float> varying_uv;
    mat<3, 3, float> varying_normal;
    mat<4, 4, float> uniform_MIT;
    mat<4, 4, float> uniform_M;
    mat<3, 3, float> varing_nrm;
    mat<4, 4, float> uniform_Mshadow;
    mat<3, 3, float> varying_tri;
    TGAColor text[3];
    GouraudShader(Matrix M, Matrix MIT, Matrix MS) : uniform_M(M), uniform_MIT(MIT), uniform_Mshadow(MS), varying_uv(), varying_tri() {}

    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        gl_Vertex       = Viewport * Projection * ModelView * gl_Vertex;     // transform it to screen coordinates
        varying_tri.set_col(nthvert, proj<3>(gl_Vertex / gl_Vertex[3]));
        gl_Vertex       = gl_Vertex / gl_Vertex[3];
        varing_verts.set_col(nthvert, model->vert(iface, nthvert));
        varying_normal.set_col(nthvert, model->normal(iface, nthvert));
        varying_intensity[nthvert]  = std::max(0.f, model->normal(iface, nthvert) * light_dir.normalize()); // get diffuse lighting intensity
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        text[nthvert]= model->diffuse(model->uv(iface, nthvert));
        varing_nrm.set_col(nthvert, proj<3>(uniform_MIT * embed<4>(model->normal(iface, nthvert), 0.f)));
        return gl_Vertex;
    }
    virtual bool ambient(Vec3f bar, TGAColor& color) {
        return false;
    }
    virtual bool fragment(Vec3f bar, TGAColor& color) {
        Vec4f tmp1 = uniform_Mshadow * embed<4>(varying_tri.col(0)); tmp1 = tmp1 / tmp1[3];
        Vec4f tmp2 = uniform_Mshadow * embed<4>(varying_tri.col(1)); tmp2 = tmp2 / tmp2[3];
        Vec4f tmp3 = uniform_Mshadow * embed<4>(varying_tri.col(2)); tmp3 = tmp3 / tmp3[3];
        Vec4f sb_p = tmp1 * bar.x + tmp2 * bar.y + tmp3 * bar.z;
        //Vec4f sb_p = uniform_Mshadow * embed<4>(varying_tri * bar); // corresponding point in the shadow buffer
        //sb_p = sb_p / sb_p[3];
        int idx = int(sb_p[0]) + int(sb_p[1]) * width; // index in the shadowbuffer array
        float shadow = .1 + .9 * (shadowbuffer[idx]-43.34 < (sb_p[2])); // magic coeff to avoid z-fighting


        Vec2f uv = varying_uv * bar;
        Vec3f n = (model->normal(uv));
        Vec3f h = ((eye - varing_verts * bar).normalize() + light_dir.normalize()).normalize();   // reflected light
        float dist2light = 3;
        float dist2eye = (eye - varing_verts * bar).norm();
        float ld = std::max(0.f, n.normalize() * light_dir.normalize());
        float ls = 3 * pow(std::max(0.f, n * h), 100);
        TGAColor c = model->diffuse(uv);
        color = c;
        for (int i = 0; i < 3; i++) color[i] = std::min<float>(20 + c[i]*shadow * (ld + .6 * ls), 255);
        //for (int i = 0; i < 3; i++) color[i] = shadow * 255;
        return false;
                            // no, we do not discard this pixel
    }
    virtual bool fragment_1(Vec3f bar, TGAColor& color) {
        float intensity                     = varying_intensity * bar;   // interpolate intensity for the current pixel
        if (intensity > .85)      intensity = 1;
        else if (intensity > .60) intensity = .80;
        else if (intensity > .45) intensity = .60;
        else if (intensity > .30) intensity = .45;
        else if (intensity > .15) intensity = .30;
        else                      intensity = 0;
        color                               = model->diffuse(varying_uv * bar) * intensity; // well duh
        return false;                              // no, we do not discard this pixel
    }
    virtual bool phong_shading(Vec3f bar, TGAColor& color) {
        Vec3f normal    = varying_normal * bar;
        float intensity = std::max(0.f,normal*light_dir);   // interpolate intensity for the current pixel
        //color = model->diffuse(varying_diffuse1 * bar) * intensity; // well duh
        color           = model->diffuse(varying_uv * bar) * intensity;
        return false;                              // no, we do not discard this pixel
    }
    virtual bool flat_shading(Vec3f bar, TGAColor& color) {
        mat<3, 2, float> vec;
        mat<2, 2, float> tmp;

        Vec3f v1 = varing_verts.col(1) - varing_verts.col(0);
        Vec3f v2 = varing_verts.col(2) - varing_verts.col(1);
        Vec2f p1 = varying_uv.col(1) - varying_uv.col(0);
        Vec2f p2 = varying_uv.col(2) - varying_uv.col(1);
        vec.set_col(0, v1);          vec.set_col(1, v2);
        tmp.set_col(0, p1);          tmp.set_col(1, p2);
        //float x1 = varying_uv[0][1] - varying_uv[0][0];
        //float x2 = varying_uv[0][2] - varying_uv[0][0];
        //float y1 = varying_uv[1][1] - varying_uv[1][0];
        //float y2 = varying_uv[1][2] - varying_uv[1][0];
        //tmp[0][0] = x1;     tmp[0][1] = x2; tmp[1][0] = y1; tmp[1][1] = y1;

        mat<3, 2, float> TB = vec * (tmp.invert());
        Vec3f T = TB.col(0).normalize(); Vec3f B = TB.col(1).normalize();
        Vec3f N = varying_normal * bar;
        T = (T - N * (T * N)).normalize();
        B = (B - N * (B * N)).normalize();
        mat<3, 3, float> TBN;
        TBN.set_col(0, T); TBN.set_col(1, B); TBN.set_col(2, N);
        Vec3f TBN_normal = model->normal(varying_uv * bar);
        for (int i = 0; i < 3; i++) {
            TBN_normal[i] = TBN_normal[i] * 2 - 1;
        }
        Vec3f n = (TBN * TBN_normal).normalize();

        float  diff = std::max(0.f, abs(n * light_dir.normalize()));
        color = model->diffuse(varying_uv * bar) * diff;
        return false;

    }
    virtual bool phong_nm(Vec3f bar, TGAColor& color) {
        Vec2f uv        = varying_uv * bar;
        Vec3f n         = proj<3>(uniform_MIT * embed<4>(model->normal(uv)));
        float intensity = std::max(0.f, n.normalize() * light_dir.normalize());
        color           = model->diffuse(uv) * intensity;
        //Vec2f uv = varying_uv * bar;                 // interpolate uv for the current pixel
        //Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        //Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        //
        //float intensity = std::max(0.f, n * l);
        //color = model->diffuse(uv) * intensity;      // well duh
        //Vec2f uv = varying_uv * bar;                 // interpolate uv for the current pixel
        //Vec3f n = proj<3>(uniform_MIT*embed<4>(model->normal(uv))).normalize();
        //Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        //float intensity = std::max(0.f, n * l);
        //color = model->diffuse(uv) * intensity;      // well duh
        return false;
    }

    virtual bool blinn_phon(Vec3f bar, TGAColor& color) {
        Vec2f uv         = varying_uv * bar;
        Vec3f n          = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        Vec3f h          =((eye-varing_verts*bar).normalize()+light_dir.normalize()).normalize();   // reflected light
        float dist2light = 3;
        float dist2eye   = (eye - varing_verts * bar).norm();
        float ld         = std::max(0.f, n.normalize() * light_dir.normalize());
        float ls         = 3*pow(std::max(0.f, n * h),100);
        TGAColor c       = model->diffuse(uv);
        color = c;
        for (int i = 0; i < 3; i++) color[i] = std::min<float>(5 + c[i] * (ld+ .6 * ls), 255);
        return false;
    }
};
//struct GouraudShader : public IShader {
//    Vec3f varying_intensity; // written by vertex shader, read by fragment shader
//    mat<2, 3, float> varying_diffuse; // written by vertex shader, read by fragment shader
//
//    virtual Vec4f vertex(int iface, int nthvert) {
//        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
//        gl_Vertex = Viewport * Projection * ModelView * gl_Vertex;     // transform it to screen coordinates
//        gl_Vertex = gl_Vertex / gl_Vertex[3];
//        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * light_dir); // get diffuse lighting intensity
//        varying_diffuse[0][nthvert] = model->uv(iface, nthvert)[0];
//        varying_diffuse[1][nthvert] = model->uv(iface, nthvert)[1];
//
//        return gl_Vertex;
//    }
//
//    virtual bool fragment(Vec3f bar, TGAColor& color) {
//        float intensity = varying_intensity * bar;   // interpolate intensity for the current pixel
//        Vec2f diffuse;
//        for (int i = 0; i < 2; i++) {
//            diffuse[i] = varying_diffuse[0][i] * bar.x + varying_diffuse[1][i] * bar.y + varying_diffuse[2][i] * bar.z;
//        }
//        color = model->diffuse(varying_diffuse * bar) * intensity; // well duh
//        return false;                              // no, we do not discard this pixel
//    }
//};

struct DepthShader : public IShader {
    Vec3f varying_intensity; // written by vertex shader, read by fragment shader
         // written by vertex shader, read by fragment shader
    mat<3, 3, float> varing_verts;

    mat<2, 3, float> varying_uv;
    mat<3, 3, float> varying_normal;
    mat<4, 4, float> uniform_MIT;
    mat<4, 4, float> uniform_M;
    mat<4, 4, float> uniform_shadow;
    mat<3, 3, float> varing_nrm;
    mat<3, 3, float> varying_tri;
    mat<4, 3, float> ambient_tri;
    DepthShader() :varying_tri(){}
    TGAColor text[3];
    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        gl_Vertex       = Viewport * Projection * ModelView * gl_Vertex;     // transform it to screen coordinates
        gl_Vertex       = gl_Vertex / gl_Vertex[3];

        varying_tri.set_col(nthvert,proj<3>(gl_Vertex/gl_Vertex[3]));
        varing_verts.set_col(nthvert, model->vert(iface, nthvert));
        varying_normal.set_col(nthvert, model->normal(iface, nthvert));
        varying_intensity[nthvert]  = std::max(0.f, model->normal(iface, nthvert) * light_dir.normalize()); // get diffuse lighting intensity
        varying_uv.set_col(nthvert, model->uv(iface, nthvert));
        text[nthvert]= model->diffuse(model->uv(iface, nthvert));
        varing_nrm.set_col(nthvert, proj<3>(uniform_MIT * embed<4>(model->normal(iface, nthvert), 0.f)));
        return gl_Vertex;
    }
    virtual bool ambient(Vec3f bar, TGAColor& color) {
    }
    virtual bool fragment(Vec3f bar, TGAColor& color) {
        Vec3f p = varying_tri * bar;
        color   = TGAColor(255, 255, 255) * (p.z / depth);
        return false;                              // no, we do not discard this pixel
    }
    virtual bool fragment_1(Vec3f bar, TGAColor& color) {
        float intensity                     = varying_intensity * bar;   // interpolate intensity for the current pixel
        if (intensity > .85)      intensity = 1;
        else if (intensity > .60) intensity = .80;
        else if (intensity > .45) intensity = .60;
        else if (intensity > .30) intensity = .45;
        else if (intensity > .15) intensity = .30;
        else                      intensity = 0;
        color                               = model->diffuse(varying_uv * bar) * intensity; // well duh
        return false;                              // no, we do not discard this pixel
    }
    virtual bool phong_shading(Vec3f bar, TGAColor& color) {
        Vec3f normal    = varying_normal * bar;
        float intensity = std::max(0.f,normal*light_dir);   // interpolate intensity for the current pixel
        //color = model->diffuse(varying_diffuse1 * bar) * intensity; // well duh
        color           = model->diffuse(varying_uv * bar) * intensity;
        return false;                              // no, we do not discard this pixel
    }
    virtual bool flat_shading(Vec3f bar, TGAColor& color) {
        mat<3, 2, float> vec;
        mat<2, 2, float> tmp;

        Vec3f v1 = varing_verts.col(1) - varing_verts.col(0);
        Vec3f v2 = varing_verts.col(2) - varing_verts.col(1);
        Vec2f p1 = varying_uv.col(1) - varying_uv.col(0);
        Vec2f p2 = varying_uv.col(2) - varying_uv.col(1);
        vec.set_col(0, v1);          vec.set_col(1, v2);
        tmp.set_col(0, p1);          tmp.set_col(1, p2);
        //float x1 = varying_uv[0][1] - varying_uv[0][0];
        //float x2 = varying_uv[0][2] - varying_uv[0][0];
        //float y1 = varying_uv[1][1] - varying_uv[1][0];
        //float y2 = varying_uv[1][2] - varying_uv[1][0];
        //tmp[0][0] = x1;     tmp[0][1] = x2; tmp[1][0] = y1; tmp[1][1] = y1;

        mat<3, 2, float> TB = vec * (tmp.invert());
        Vec3f T = TB.col(0).normalize(); Vec3f B = TB.col(1).normalize();
        Vec3f N = varying_normal * bar;
        T = (T - N * (T * N)).normalize();
        B = (B - N * (B * N)).normalize();
        mat<3, 3, float> TBN;
        TBN.set_col(0, T); TBN.set_col(1, B); TBN.set_col(2, N);
        Vec3f TBN_normal = model->normal(varying_uv * bar);
        for (int i = 0; i < 3; i++) {
            TBN_normal[i] = TBN_normal[i] * 2 - 1;
        }
        Vec3f n = (TBN * TBN_normal).normalize();

        float  diff = std::max(0.f, abs(n * light_dir.normalize()));
        color = model->diffuse(varying_uv * bar) * diff;
        return false;

    }
    virtual bool phong_nm(Vec3f bar, TGAColor& color) {
        Vec2f uv        = varying_uv * bar;
        Vec3f n         = proj<3>(uniform_MIT * embed<4>(model->normal(uv)));
        float intensity = std::max(0.f, n.normalize() * light_dir.normalize());
        color           = model->diffuse(uv) * intensity;
        //Vec2f uv = varying_uv * bar;                 // interpolate uv for the current pixel
        //Vec3f n = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        //Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        //
        //float intensity = std::max(0.f, n * l);
        //color = model->diffuse(uv) * intensity;      // well duh
        //Vec2f uv = varying_uv * bar;                 // interpolate uv for the current pixel
        //Vec3f n = proj<3>(uniform_MIT*embed<4>(model->normal(uv))).normalize();
        //Vec3f l = proj<3>(uniform_M * embed<4>(light_dir)).normalize();
        //float intensity = std::max(0.f, n * l);
        //color = model->diffuse(uv) * intensity;      // well duh
        return false;
    }

    virtual bool blinn_phon(Vec3f bar, TGAColor& color) {
        Vec2f uv         = varying_uv * bar;
        Vec3f n          = proj<3>(uniform_MIT * embed<4>(model->normal(uv))).normalize();
        Vec3f h          =((eye-varing_verts*bar).normalize()+light_dir.normalize()).normalize();   // reflected light
        float dist2light = 3;
        float dist2eye   = (eye - varing_verts * bar).norm();
        float ld         = std::max(0.f, n.normalize() * light_dir.normalize());
        float ls         = 0.8*pow(std::max(0.f, n * h),100);
        TGAColor c       = model->diffuse(uv);
        color = c;
        for (int i = 0; i < 3; i++) color[i] = std::min<float>(5 + c[i] * (ld+ .6 * ls), 255);
        return false;
    }
};
//struct GouraudShader : public IShader {
//    Vec3f varying_intensity; // written by vertex shader, read by fragment shader
//    mat<2, 3, float> varying_diffuse; // written by vertex shader, read by fragment shader
//
//    virtual Vec4f vertex(int iface, int nthvert) {
//        Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
//        gl_Vertex = Viewport * Projection * ModelView * gl_Vertex;     // transform it to screen coordinates
//        gl_Vertex = gl_Vertex / gl_Vertex[3];
//        varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * light_dir); // get diffuse lighting intensity
//        varying_diffuse[0][nthvert] = model->uv(iface, nthvert)[0];
//        varying_diffuse[1][nthvert] = model->uv(iface, nthvert)[1];
//
//        return gl_Vertex;
//    }
//
//    virtual bool fragment(Vec3f bar, TGAColor& color) {
//        float intensity = varying_intensity * bar;   // interpolate intensity for the current pixel
//        Vec2f diffuse;
//        for (int i = 0; i < 2; i++) {
//            diffuse[i] = varying_diffuse[0][i] * bar.x + varying_diffuse[1][i] * bar.y + varying_diffuse[2][i] * bar.z;
//        }
//        color = model->diffuse(varying_diffuse * bar) * intensity; // well duh
//        return false;                              // no, we do not discard this pixel
//    }
//};
struct Zshader : public IShader {
    mat<4, 3, float>varying_tri;
    virtual Vec4f vertex(int iface, int nthvert) {
        Vec4f gl_Vertex =Projection*ModelView* embed<4>(model->vert(iface, nthvert)); // read the vertex from .obj file
        varying_tri.set_col(nthvert, gl_Vertex);
        return gl_Vertex;
    }
    virtual bool ambient(Vec3f gl_FragCoord, Vec3f bar, TGAColor& color) {
        color = TGAColor(0, 0, 0);
        return false;
    }
    virtual bool fragment(Vec3f bar, TGAColor& color) {
        return false;                              // no, we do not discard this pixel
    }
    virtual bool fragment_1(Vec3f bar, TGAColor& color) {

        return false;                              // no, we do not discard this pixel
    }
    virtual bool phong_shading(Vec3f bar, TGAColor& color) {

        return false;                              // no, we do not discard this pixel
    }
    virtual bool flat_shading(Vec3f bar, TGAColor& color) {

        return false;

    }
    virtual bool phong_nm(Vec3f bar, TGAColor& color) {
        return false;
    }
    virtual bool blinn_phon(Vec3f bar, TGAColor& color) {
        return false;
    }
};
float max_elevation_angle(float* zbuffer, Vec2f p, Vec2f dir) {
    float maxangle = 0;
    for (float t = 0.; t < 1000.; t += 1.) {
        Vec2f cur = p + dir * t;
        if (cur.x >= width || cur.y >= height || cur.x < 0 || cur.y < 0) return maxangle;

        float distance = sqrt((p - cur).x*(p-cur).x+ (p - cur).y * (p - cur).y);
        if (distance < 1.f) continue;
        float elevation = zbuffer[int(cur.x) + int(cur.y) * width] - zbuffer[int(p.x) + int(p.y) * width];
        maxangle = std::max(maxangle, atanf(elevation / distance));
    }
    return maxangle;
}


int main(int argc, char** argv) {
    if (2==argc) {
        model = new Model(argv[1]);
    } else {
        model = new Model("obj/diablo3_pose.obj");
    }
    light_dir.normalize();

    float *zbuffer = new float[width*height];
    shadowbuffer = new float[width * height];
    for (int i = width * height; --i; ) {
        zbuffer[i] = shadowbuffer[i] = -std::numeric_limits<float>::max();
    }
    TGAImage image  (width, height, TGAImage::RGB);

    lookat(eye, center, up);
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    projection(-1.f / (eye.z / abs(eye.z) * sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z)));

    Zshader zshader;
    for (int i = 0; i < model->nfaces(); i++) {
        for (int j = 0; j < 3; j++) {
            zshader.vertex(i, j);
        }
        triangle(zshader.varying_tri, zshader, image, zbuffer);
    }
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (zbuffer[x + y * width] < -1e5) continue;
            float total = 0;
            for (float a = 0; a <M_PI * 2 - 1e-4; a += M_PI / 4) {
                total += M_PI / 2 - max_elevation_angle(zbuffer, Vec2f(x, y), Vec2f(cos(a), sin(a)));
            }
            total /= (M_PI / 2) * 8;
            total = pow(total, 100.f);
            image.set(x, y, TGAColor(total * 255, total * 255, total * 255));
        }
    }
    image.flip_vertically();
    image.write_tga_file("mbient.tga");
    //{ÒõÓ°Ó³Éä
    //    TGAImage depth(width, height, TGAImage::RGB);
    //    lookat(light_dir, center, up);
    //    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    //    projection(0);

    //    DepthShader depthshader;
    //    Vec4f screen_coords[3];
    //    for (int i = 0; i < model->nfaces(); i++) {
    //        for (int j = 0; j <3; j++)
    //        {
    //            screen_coords[j] = depthshader.vertex(i, j);
    //        }
    //        triangle(screen_coords, depthshader, depth, shadowbuffer);
    //    }
    //    depth.flip_vertically();
    //    depth.write_tga_file("depth.tga");
    //}
    //Matrix M = Viewport * Projection * ModelView;
    //{   TGAImage frame(width, height, TGAImage::RGB);
    //    lookat(eye, center, up);
    //    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    //    projection(-1.f / (eye.z / abs(eye.z) * sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z)));


    //    GouraudShader shader(ModelView,(ModelView).invert_transpose(), M * (Viewport * Projection * ModelView).invert());
    //    
    //    light_dir = proj<3>(shader.uniform_M * embed<4>(light_dir)).normalize();

    //    Vec4f screen_coords[3];
    //    for (int i=0; i<model->nfaces(); i++) {
    //        for (int j=0; j<3; j++) {
    //            screen_coords[j] = shader.vertex(i, j);
    //        }
    //        triangle(screen_coords, shader, image, zbuffer);
    //    }

    //    image.  flip_vertically(); // to place the origin in the bottom left corner of the image
    //    image.  write_tga_file("framebuffer.tga");
    //    
    //}


    //lookat(eye, center, up);
    //viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);
    //projection(-1.f / (eye.z / abs(eye.z) * sqrt(eye.x * eye.x + eye.y * eye.y + eye.z * eye.z)));


    //GouraudShader shader(ModelView,ModelView.invert_transpose(),ModelView);
    ////shader.uniform_M = ModelView;
    ////for (int tmp = 0; tmp < 4; tmp++) {
    ////    shader.uniform_M[2][tmp] = shader.uniform_M[2][tmp];
    ////}
    ////shader.uniform_MIT = shader.uniform_M.invert_transpose();
    ////light_dir = proj<3>(shader.uniform_M * embed<4>(light_dir)).normalize();

    //
    //for (int i=0; i<model->nfaces(); i++) {
    //    Vec4f screen_coords[3];
    //    for (int j=0; j<3; j++) {
    //        screen_coords[j] = shader.vertex(i, j);
    //    }
    //    triangle(screen_coords, shader, image, zbuffer);
    //}

    //image.  flip_vertically(); // to place the origin in the bottom left corner of the image
    //image.  write_tga_file("output.tga");

    delete model;
    return 0;
}
