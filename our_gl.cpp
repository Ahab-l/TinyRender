#include <cmath>
#include <limits>
#include <cstdlib>
#include "our_gl.h"

Matrix ModelView;
Matrix Viewport;
Matrix Projection;

IShader::~IShader() {}

void viewport(int x, int y, int w, int h) {
    Viewport = Matrix::identity();
    Viewport[0][3] = x+w/2.f;
    Viewport[1][3] = y+h/2.f;
    Viewport[2][3] = depth/2.f;
    Viewport[0][0] = w/2.f;
    Viewport[1][1] = h/2.f;
    Viewport[2][2] = depth/2.f;
}

void projection(float coeff) {
    Projection = Matrix::identity();
    Projection[3][2] = coeff;
}

void lookat(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f z = (center-eye).normalize();
    Vec3f x = cross(z,up).normalize();
    Vec3f y = cross(x,z).normalize();
    ModelView = Matrix::identity();
    for (int i=0; i<3; i++) {
        ModelView[0][i] = x[i];
        ModelView[1][i] = y[i];
        ModelView[2][i] = -z[i];

    }
}

Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P) {
    Vec3f s[2];
    for (int i=2; i--; ) {
        s[i][0] = C[i]-A[i];
        s[i][1] = B[i]-A[i];
        s[i][2] = A[i]-P[i];
    }
    Vec3f u = cross(s[0], s[1]);
    if (std::abs(u[2])>1e-2) // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
        return Vec3f(1.f-(u.x+u.y)/u.z, u.y/u.z, u.x/u.z);
    return Vec3f(-1,1,1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}
Vec3f barycentric_2D(int x, int y, Vec4f tri[]) {
    float denominator = (tri[0][1] - tri[2][1]) * (tri[1][0] - tri[2][0]) + (tri[1][1] - tri[2][1]) * (tri[2][0] - tri[0][0]);
    float b1 = ((y - tri[2][1]) * (tri[1][0] - tri[2][0]) + (tri[1][1] - tri[2][1]) * (tri[2][0] - x)) / denominator;
    float b2 = ((y - tri[0][1]) * (tri[2][0] - tri[0][0]) + (tri[2][1] - tri[0][1]) * (tri[0][0] - x)) / denominator;
    float b3 = ((y - tri[1][1]) * (tri[0][0] - tri[1][0]) + (tri[0][1] - tri[1][1]) * (tri[1][0] - x)) / denominator;
    float z = 1.f / ((b1 / tri[0][2]) + (b2 / tri[1][2]) + (b3 / tri[2][2]));

    float alpha = z / tri[0][2] * b1;
    float beta = z / tri[1][2] * b2;
    float gamma = z / tri[2][2] * b3;
    return Vec3f(alpha, beta, gamma);
}
Vec2f* foundingbox_2D(Vec4f t[]) {//找出2D 三角的包围盒
    int MaxX = std::max(t[0][0], std::max(t[1][0], t[2][0]));
    int MaxY = std::max(t[0][1], std::max(t[1][1], t[2][1]));
    int MinX = std::min(t[0][0], std::min(t[1][0], t[2][0]));
    int MinY = std::min(t[0][1], std::min(t[1][1], t[2][1]));
    Vec2f* TAB = new Vec2f[2];
    TAB[0] = Vec2f(MinX, MinY);
    TAB[1] = Vec2f(MaxX, MaxY);
    return TAB;
}

void triangle(Vec4f *pts, IShader &shader, TGAImage &image, float *zbuffer) {
    Vec2f* TAB = foundingbox_2D(pts);



    TGAColor color;
    for (int x = (TAB[0].x > 0 ? TAB[0].x : 0); x <= (TAB[1].x < image.get_width() ? TAB[1].x : image.get_width()); x++) {
        for (int y = (TAB[0].y > 0 ? TAB[0].y : 0); y <= (TAB[1].y < image.get_height() ? TAB[1].y : image.get_width()); y++) {
            //Vec3f c = barycentric(proj<2>(pts[0]/pts[0][3]), proj<2>(pts[1]/pts[1][3]), proj<2>(pts[2]/pts[2][3]), proj<2>(P));
            Vec3f c = barycentric_2D(x, y, pts);
            float z = pts[0][2]*c.x + pts[1][2]*c.y + pts[2][2]*c.z;
            //float w = pts[0][3]*c.x + pts[1][3]*c.y + pts[2][3]*c.z;
            //int frag_depth = std::max(0, std::min(255, int(z/w+.5)));
            if (c.x<0 || c.y<0 || c.z<0 || zbuffer[x+y*image.get_width()]>z) continue;
            bool discard = shader.fragment(c, color);
            if (!discard) {
                zbuffer[x+y*image.get_width()]=z ;
                image.set(x, y, color);
            }
        }
    }
}
void triangle(mat<4, 3, float>& clipc, IShader& shader, TGAImage& image, float* zbuffer) {
    mat<3, 4, float> pts = (Viewport * clipc).transpose(); // transposed to ease access to each of the points
    mat<3, 2, float> pts2;
    for (int i = 0; i < 3; i++) pts2[i] = proj<2>(pts[i] / pts[i][3]);

    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    Vec2f clamp(image.get_width() - 1, image.get_height() - 1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            bboxmin[j] = std::max(0.f, std::min(bboxmin[j], pts2[i][j]));
            bboxmax[j] = std::min(clamp[j], std::max(bboxmax[j], pts2[i][j]));
        }
    }
    Vec2i P;
    TGAColor color;
    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) {
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) {
            Vec3f bc_screen = barycentric(pts2[0], pts2[1], pts2[2], P);
            Vec3f bc_clip = Vec3f(bc_screen.x / pts[0][3], bc_screen.y / pts[1][3], bc_screen.z / pts[2][3]);
            bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);//重心坐标插值
            float frag_depth = clipc[2] * bc_clip;
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z<0 || zbuffer[P.x + P.y * image.get_width()]>frag_depth) continue;
            bool discard = shader.ambient(Vec3f(P.x, P.y, frag_depth), bc_clip, color);
            if (!discard) {
                zbuffer[P.x + P.y * image.get_width()] = frag_depth;
                image.set(P.x, P.y, color);
            }
        }
    }
}
