#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <cstdlib>

//Constant
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(const double& degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(const double& min, const double& max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_double();
}

//vec3
class vec3 {
public:
    float e[3];

    vec3() : e{ 0, 0, 0 } {}
    vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    float x() const { return e[0]; }
    float y() const { return e[1]; }
    float z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    float operator[](int i) const { return e[i]; }
    float& operator[](int i) { return e[i]; }

    vec3(const vec3& other) : e{ other[0], other[1], other[2] } {}
    vec3& operator=(const vec3& other) { e[0] = other[0]; e[1] = other[1]; e[2] = other[2]; return *this; }

    vec3& operator+=(const vec3& other) { e[0] += other[0]; e[1] += other[1]; e[2] += other[2]; return *this; }
    vec3& operator*=(const float& t) { e[0] *= t; e[1] *= t; e[2] *= t; return *this; }
    vec3& operator/=(const float& t) { return *this *= 1 / t; }
    inline vec3 operator+(const vec3& other) const { return vec3(e[0] + other[0], e[1] + other[1], e[2] + other[2]); }
    inline vec3 operator+(const float& t) const { return vec3(e[0] + t, e[1] + t, e[2] + t); }
    inline vec3 operator-(const vec3& other) const { return vec3(e[0] - other[0], e[1] - other[1], e[2] - other[2]); }
    inline vec3 operator-(const float& t) const { return vec3(e[0] - t, e[1] - t, e[2] - t); }
    inline vec3 operator*(const float& t) const { return vec3(e[0] * t, e[1] * t, e[2] * t); }
    inline vec3 operator*(const vec3& other) const { return vec3(e[0] * other[0], e[1] * other[1], e[2] * other[2]); }
    inline vec3 operator/(const float& t) const { return *this * (1 / t); }

    float length() { return sqrt(length_squared()); }
    float length_squared() { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    inline static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    inline static vec3 random(const double& min, const double& max) {
        return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
    }
};

vec3 random_in_unit_sphere() {
    while (true) {
        vec3 p = vec3::random(-1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

vec3 random_unit_vector() {
    float a = random_double(0, 2 * pi);
    float z = random_double(-1, 1);
    float r = sqrt(1 - z * z);
    return vec3(r * cos(a), r * sin(a), z);
}

vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

using point3 = vec3; //3D point
using color = vec3;  //RGB color

std::ostream& operator<<(std::ostream& out, const vec3& vec) { return out << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]"; }

inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

float clamp(const float& x, const float& min, const float& max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

void write_color(std::ostream& out, const color& pixel_color, const int& samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples.
    /*auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);*/

    r = sqrt(r);
    g = sqrt(g);
    b = sqrt(b);

    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

//ray
class ray {
public:
    point3 orig;
    vec3 dir;

    ray() {}
    ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    point3 origin() const { return orig; }
    vec3 direction() const { return dir; }

    point3 at(const float& t) const {
        return orig + (dir * t);
    }
};

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

struct hit_record;

//Material
class material {
public:
    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const = 0;
};

//hittable

struct hit_record {
    point3 p;
    vec3 normal;
    std::shared_ptr<material> mat_ptr;
    float t;
    bool front_face;

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual bool hit(const ray& r, const float& t_min, const float& t_max, hit_record& rec) const = 0;
};

class sphere : public hittable {
public:
    point3 center;
    float radius;
    std::shared_ptr<material> mat_ptr;
    sphere(point3 cen, float r) : center(cen), radius(r), mat_ptr(0) {}
    sphere(point3 cen, float r, std::shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {}
    virtual bool hit(const ray& r, const float& t_min, const float& t_max, hit_record& rec) const override;
};

bool sphere::hit(const ray& r, const float& t_min, const float& t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = (dot(oc, r.dir));
    float c = oc.length_squared() - radius * radius;
    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return false;
    }
    else {
        float root = sqrt(discriminant);
        float temp = (-half_b - root) / a;
        if (temp >= t_min && temp <= t_max) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp >= t_min && temp <= t_max) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            return true;
        }
        return false;
    }
}

class hittable_list : public hittable {
public:
    std::vector<std::shared_ptr<hittable>> objects;
    hittable_list() {}
    hittable_list(std::shared_ptr<hittable> obj) { add(obj); }
    void clear() { objects.clear(); }
    void add(std::shared_ptr<hittable> obj) { objects.push_back(obj); }
    virtual bool hit(const ray& r, const float& t_min, const float& t_max, hit_record& rec) const override;
};

bool hittable_list::hit(const ray& r, const float& t_min, const float& t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (const auto& obj : objects) {
        if (obj->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
};


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
//Camera

class camera {
private:
    point3 origin;
    vec3 horizontal;
    vec3 vertical;
    vec3 lower_left_corner;
    float lens_radius;
    vec3 u, v, w;
public:
    camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist) {
        float theta = degrees_to_radians(vfov);
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = viewport_height * aspect_ratio;
        float focal_length = 1.0;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = u * viewport_width * focus_dist;
        vertical = v * viewport_height * focus_dist;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w * focus_dist;
        lens_radius = aperture / 2;
    }

    ray getray(const double& s, const double& t) const {
        vec3 rd = random_in_unit_disk() * lens_radius;
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + horizontal * s + vertical * t - origin - offset);
    }
};


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

class lambertian : public material {
public:
    color albedo;

    lambertian(const color& a) : albedo(a) {}
    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        vec3 scatter_direction = rec.normal + random_unit_vector();
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

vec3 reflect(const vec3& v, const vec3& n) {
    return v - (n * dot(v, n)) * 2;
}

class metal : public material {
public:
    color albedo;
    float fuzz;
    metal(const color& a, const float& f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + random_in_unit_sphere() * fuzz);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

};

vec3 refract(const vec3& uv, const vec3& n, const float& etai_over_etat) {
    float cos_theta = dot(-uv, n);
    vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    vec3 r_out_parallel = n * -sqrt(fabs(1 - r_out_perp.length_squared()));
    return r_out_parallel + r_out_perp;
};

double schlick(const double& cosine, const double& ref_idx) {
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
};

class dielectric : public material {
public:
    float ref_idx;
    dielectric(float idx) : ref_idx(idx) {}
    virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
        attenuation = color(1.0, 1.0, 1.0);
        float etai_over_etat = rec.front_face ? (1.0 / ref_idx) : ref_idx;

        vec3 unit_direction = unit_vector(r_in.direction());

        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        if (etai_over_etat * sin_theta > 1.0) {
            vec3 reflected = reflect(unit_direction, rec.normal);
            scattered = ray(rec.p, reflected);
            return true;
        }

        double reflect_prob = schlick(cos_theta, etai_over_etat);
        if (random_double() < reflect_prob)
        {
            vec3 reflected = reflect(unit_direction, rec.normal);
            scattered = ray(rec.p, reflected);
            return true;
        }

        vec3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
        scattered = ray(rec.p, refracted);
        return true;
    }
};