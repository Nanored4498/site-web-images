#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cassert>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

struct mat4 {
	double _m[16];

	static const mat4 Id;

	inline const double& operator()(int i, int j) const { return _m[4*i+j]; }
	inline double& operator()(int i, int j) { return _m[4*i+j]; }

	inline mat4& operator+=(const mat4 &m) { for(int i = 0; i < 16; ++i) _m[i] += m._m[i]; return *this; }
	inline mat4 operator+(const mat4 &m) const { return mat4(*this) += m; }
	inline mat4& operator-=(const mat4 &m) { for(int i = 0; i < 16; ++i) _m[i] -= m._m[i]; return *this; }
	inline mat4 operator-(const mat4 &m) const { return mat4(*this) -= m; }
	inline mat4 operator*(const mat4 &m) const {
		mat4 res{0.};
		for(int i = 0; i < 16; ++i) assert(res._m[i] == 0.);
		for(int i = 0; i < 4; ++i) for(int j = 0; j < 4; ++j) for(int k = 0; k < 4; ++k)
			res(i,j) += operator()(i,k) * m(k,j);
		return res;
	}

	inline mat4 T() const {
		mat4 res;
		for(int i = 0; i < 4; ++i) for(int j = 0; j < 4; ++j) res(i,j)=operator()(j,i);
		return res;
	}

	friend mat4 inv(const mat4 &mm) {
		mat4 res;
		double* inv = res._m;
		const double* m = mm._m;

		inv[0] = m[5]  * m[10] * m[15] - 
				m[5]  * m[11] * m[14] - 
				m[9]  * m[6]  * m[15] + 
				m[9]  * m[7]  * m[14] +
				m[13] * m[6]  * m[11] - 
				m[13] * m[7]  * m[10];

		inv[4] = -m[4]  * m[10] * m[15] + 
				m[4]  * m[11] * m[14] + 
				m[8]  * m[6]  * m[15] - 
				m[8]  * m[7]  * m[14] - 
				m[12] * m[6]  * m[11] + 
				m[12] * m[7]  * m[10];

		inv[8] = m[4]  * m[9] * m[15] - 
				m[4]  * m[11] * m[13] - 
				m[8]  * m[5] * m[15] + 
				m[8]  * m[7] * m[13] + 
				m[12] * m[5] * m[11] - 
				m[12] * m[7] * m[9];

		inv[12] = -m[4]  * m[9] * m[14] + 
				m[4]  * m[10] * m[13] +
				m[8]  * m[5] * m[14] - 
				m[8]  * m[6] * m[13] - 
				m[12] * m[5] * m[10] + 
				m[12] * m[6] * m[9];

		inv[1] = -m[1]  * m[10] * m[15] + 
				m[1]  * m[11] * m[14] + 
				m[9]  * m[2] * m[15] - 
				m[9]  * m[3] * m[14] - 
				m[13] * m[2] * m[11] + 
				m[13] * m[3] * m[10];

		inv[5] = m[0]  * m[10] * m[15] - 
				m[0]  * m[11] * m[14] - 
				m[8]  * m[2] * m[15] + 
				m[8]  * m[3] * m[14] + 
				m[12] * m[2] * m[11] - 
				m[12] * m[3] * m[10];

		inv[9] = -m[0]  * m[9] * m[15] + 
				m[0]  * m[11] * m[13] + 
				m[8]  * m[1] * m[15] - 
				m[8]  * m[3] * m[13] - 
				m[12] * m[1] * m[11] + 
				m[12] * m[3] * m[9];

		inv[13] = m[0]  * m[9] * m[14] - 
				m[0]  * m[10] * m[13] - 
				m[8]  * m[1] * m[14] + 
				m[8]  * m[2] * m[13] + 
				m[12] * m[1] * m[10] - 
				m[12] * m[2] * m[9];

		inv[2] = m[1]  * m[6] * m[15] - 
				m[1]  * m[7] * m[14] - 
				m[5]  * m[2] * m[15] + 
				m[5]  * m[3] * m[14] + 
				m[13] * m[2] * m[7] - 
				m[13] * m[3] * m[6];

		inv[6] = -m[0]  * m[6] * m[15] + 
				m[0]  * m[7] * m[14] + 
				m[4]  * m[2] * m[15] - 
				m[4]  * m[3] * m[14] - 
				m[12] * m[2] * m[7] + 
				m[12] * m[3] * m[6];

		inv[10] = m[0]  * m[5] * m[15] - 
				m[0]  * m[7] * m[13] - 
				m[4]  * m[1] * m[15] + 
				m[4]  * m[3] * m[13] + 
				m[12] * m[1] * m[7] - 
				m[12] * m[3] * m[5];

		inv[14] = -m[0]  * m[5] * m[14] + 
				m[0]  * m[6] * m[13] + 
				m[4]  * m[1] * m[14] - 
				m[4]  * m[2] * m[13] - 
				m[12] * m[1] * m[6] + 
				m[12] * m[2] * m[5];

		inv[3] = -m[1] * m[6] * m[11] + 
				m[1] * m[7] * m[10] + 
				m[5] * m[2] * m[11] - 
				m[5] * m[3] * m[10] - 
				m[9] * m[2] * m[7] + 
				m[9] * m[3] * m[6];

		inv[7] = m[0] * m[6] * m[11] - 
				m[0] * m[7] * m[10] - 
				m[4] * m[2] * m[11] + 
				m[4] * m[3] * m[10] + 
				m[8] * m[2] * m[7] - 
				m[8] * m[3] * m[6];

		inv[11] = -m[0] * m[5] * m[11] + 
				m[0] * m[7] * m[9] + 
				m[4] * m[1] * m[11] - 
				m[4] * m[3] * m[9] - 
				m[8] * m[1] * m[7] + 
				m[8] * m[3] * m[5];

		inv[15] = m[0] * m[5] * m[10] - 
				m[0] * m[6] * m[9] - 
				m[4] * m[1] * m[10] + 
				m[4] * m[2] * m[9] + 
				m[8] * m[1] * m[6] - 
				m[8] * m[2] * m[5];

    	double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
		if(abs(det) <= 1e-40 || isnan(det)) cerr << det << ' ' << mm << endl;
		assert(abs(det) > 1e-40);
		det = 1.0 / det;
		for(int i = 0; i < 16; ++i) inv[i] *= det;
		return res;
	}

	friend double det(const mat4 &mm) {
		double inv[4];
		const double* m = mm._m;
		inv[0] = m[5]  * m[10] * m[15] - 
				m[5]  * m[11] * m[14] - 
				m[9]  * m[6]  * m[15] + 
				m[9]  * m[7]  * m[14] +
				m[13] * m[6]  * m[11] - 
				m[13] * m[7]  * m[10];
		inv[1] = -m[4]  * m[10] * m[15] + 
				m[4]  * m[11] * m[14] + 
				m[8]  * m[6]  * m[15] - 
				m[8]  * m[7]  * m[14] - 
				m[12] * m[6]  * m[11] + 
				m[12] * m[7]  * m[10];
		inv[2] = m[4]  * m[9] * m[15] - 
				m[4]  * m[11] * m[13] - 
				m[8]  * m[5] * m[15] + 
				m[8]  * m[7] * m[13] + 
				m[12] * m[5] * m[11] - 
				m[12] * m[7] * m[9];
		inv[3] = -m[4]  * m[9] * m[14] + 
				m[4]  * m[10] * m[13] +
				m[8]  * m[5] * m[14] - 
				m[8]  * m[6] * m[13] - 
				m[12] * m[5] * m[10] + 
				m[12] * m[6] * m[9];
    	return m[0] * inv[0] + m[1] * inv[1] + m[2] * inv[2] + m[3] * inv[3];
	}

	inline friend double abs(const mat4 &m) {
		double n2 = 0.; for(int i = 0; i < 16; ++i) n2 += m._m[i]*m._m[i];
		return sqrt(n2);
	}

	friend ostream& operator<<(ostream &stream, const mat4 &m) {
		stream << '[';
		for(int i = 0; i < 4; ++i) {
			if(i) stream << " ; ";
			stream << '(' << m(i,0);
			for(int j = 1; j < 4; ++j) stream << ',' << m(i,j);
			stream << ')';
		}
		return stream << ']';
	}
};
const mat4 mat4::Id = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};

struct Quaternion {
	double a, b, c, d;
	Quaternion() = default;
	inline constexpr Quaternion(double a, double b=0., double c=0., double d=0.): a(a), b(b), c(c), d(d) {}

	inline const double& operator[](int i) const { return reinterpret_cast<const double*>(this)[i]; }
	inline double& operator[](int i) { return reinterpret_cast<double*>(this)[i]; }

	inline Quaternion operator-() const {
		return Quaternion(-a, -b, -c, -d);
	}
	inline friend Quaternion operator*(double x, const Quaternion &q) {
		return Quaternion(x*q.a, x*q.b, x*q.c, x*q.d);
	}
	inline Quaternion operator/(double x) {
		return Quaternion(a/x, b/x, c/x, d/x);
	}

	inline Quaternion operator+(const Quaternion &q) const {
		return Quaternion(a+q.a, b+q.b, c+q.c, d+q.d);
	}
	inline Quaternion operator-(const Quaternion &q) const {
		return Quaternion(a-q.a, b-q.b, c-q.c, d-q.d);
	}
	inline Quaternion operator*(const Quaternion &q) const {
		return Quaternion(a*q.a - b*q.b - c*q.c - d*q.d,
						  b*q.a + a*q.b - d*q.c + c*q.d,
						  c*q.a + d*q.b + a*q.c - b*q.d,
						  d*q.a - c*q.b + b*q.c + a*q.d);
	}

	inline mat4 matLeft() const {
		return mat4{a, -b, -c, -d,
					b,  a, -d,  c,
					c,  d,  a, -b,
					d, -c,  b,  a };
	}
	inline mat4 matRight() const {
		return mat4{a, -b, -c, -d,
					b,  a,  d, -c,
					c, -d,  a,  b,
					d,  c, -b,  a };
	}

	inline Quaternion& operator+=(const Quaternion &q) {
		a += q.a; b += q.b; c += q.c; d += q.d;
		return *this;
	}
	inline Quaternion& operator-=(const Quaternion &q) {
		a -= q.a; b -= q.b; c -= q.c; d -= q.d;
		return *this;
	}
	inline Quaternion& operator*=(const Quaternion &q) { return *this = *this * q; }

	inline friend double abs(const Quaternion &q) {
		return sqrt(q.a*q.a + q.b*q.b + q.c*q.c + q.d*q.d);
	}
	inline friend Quaternion conj(const Quaternion &q) {
		return Quaternion(q.a, -q.b, -q.c, -q.d);
	}
	inline friend Quaternion inv(const Quaternion &q) {
		const double n2 = q.a*q.a + q.b*q.b + q.c*q.c + q.d*q.d;
		return Quaternion(q.a/n2, -q.b/n2, -q.c/n2, -q.d/n2);
	}

	inline friend double dot(const Quaternion &x, const Quaternion &y) {
		return x.a*y.a + x.b*y.b + x.c*y.c + x.d*y.d;
	}

	inline friend ostream& operator<<(ostream &stream, const Quaternion &q) {
		return stream << '(' << q.a << ',' << q.b << ',' << q.c << ',' << q.d << ')';
	}
};

struct vec3 {
	double x, y, z;
	vec3(double x=0., double y=0., double z=0.): x(x), y(y), z(z) {}

	inline vec3& operator*=(double a) { x*=a; y*=a; z*=a; return *this; }
	inline friend vec3 operator*(double a, const vec3 &v) { return vec3(v) *= a; }
	inline friend vec3 operator*(const vec3 &v, double a) { return vec3(v) *= a; }
	inline vec3& operator/=(double a) { x/=a; y/=a; z/=a; return *this; }
	inline friend vec3 operator/(double a, const vec3 &v) { return vec3(v) /= a; }
	inline friend vec3 operator/(const vec3 &v, double a) { return vec3(v) /= a; }

	inline vec3& operator+=(const vec3 &v) { x+=v.x; y+=v.y; z+=v.z; return *this; }
	inline vec3 operator+(const vec3 &v) const { return vec3(*this) += v; }
	inline vec3& operator-=(const vec3 &v) { x-=v.x; y-=v.y; z-=v.z; return *this; }
	inline vec3 operator-(const vec3 &v) const { return vec3(*this) -= v; }
	inline vec3& operator*=(const vec3 &v) { x*=v.x; y*=v.y; z*=v.z; return *this; }
	inline vec3 operator*(const vec3 &v) const { return vec3(*this) *= v; }

	inline double friend dot(const vec3 &a, const vec3 &b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
	inline vec3 friend cross(const vec3 &a, const vec3 &b) {
		return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}

	inline double norm() const { return sqrt(x*x+y*y+z*z); }
	inline vec3& normalize() { return *this /= norm(); }

	inline friend ostream& operator<<(ostream &stream, const vec3 &v) {
		return stream << '(' << v.x << ',' << v.y << ',' << v.z << ')';
	}
};

inline Quaternion operator*(const mat4 &m, const Quaternion &q) {
	Quaternion res(0.);
	for(int i = 0; i < 4; ++i) for(int j = 0; j < 4; ++j) res[i] += m(i, j) * q[j];
	return res;
}

Quaternion basis[3] {
	Quaternion(1., 0., 0., 0.),
	Quaternion(0., 1., 0., 0.),
	Quaternion(0., 0., 1., 0.),
};

inline Quaternion v2q(const vec3 &v) { return v.x*basis[0] + v.y*basis[1] + v.z*basis[2]; }

#define FUN 2

const double EPS = 1e-5;
const int    MAX = 80;
double getDist0(Quaternion q) {
	double dq = 1.;
	for(int k = 0; k < MAX; ++k) {
		#if FUN == 1
		if(const double d = abs(q - 1.); d < EPS) return .25 * (1. - pow(d, pow(2., 1-k))) * d * pow(2., k) / dq;
		const double nu = sqrt(3./4.);
		const double nq = nu / sqrt(q.b*q.b + q.c*q.c + q.d*q.d);
		const Quaternion proj(-.5, nq*q.b, nq*q.c, nq*q.d);
		if(const double d = abs(q - proj); d < EPS) return .25 * (1. - pow(d, pow(2., 1-k))) * d * pow(2., k) / dq;
		const Quaternion iq = inv(q);
		const Quaternion step = (1./3.) * (q - iq*iq);
		dq *= 2. * abs(iq) * abs(step);
		q -= step;
		#else
		const Quaternion c(-0.2,0.8,0,0);
		const double d = abs(q);
		// if(d > 20.) return .5 * d * log(d) / dq;
		if(d > 20.) return (1. - pow(d, -pow(2., 1-k))) * pow(2., k-2) * d / dq;
		q = q*q + c;
		dq *= 2. * d;
		#endif
	}
	// cerr << q << endl;
	return 0.;
}
Quaternion getLim(Quaternion q) {
	double res = 20.;
	Quaternion mid(100, 100, 100);
	for(int k = 0; k < MAX; ++k) {
		#if FUN == 1
		if(abs(q - 1.) < EPS) return 1.;
		const double nu = sqrt(3./4.);
		const double nq = nu / sqrt(q.b*q.b + q.c*q.c + q.d*q.d);
		const Quaternion proj(-.5, nq*q.b, nq*q.c, nq*q.d);
		if(abs(q - proj) < EPS) return proj;
		const Quaternion iq = inv(q);
		q -= (1./3.) * (q - iq*iq);
		#else
		const Quaternion c(-0.2,0.8,0,0);
		const Quaternion r1(1.3270060421587897,-0.48367240335494177), r2(-0.3270060421587897,0.48367240335494177);
		res = min({res, abs(q-r1), abs(q-r2)});
		mid.a = min(mid.a, sqrt(pow(q.c-.2, 2.) + pow(q.b-.2, 2.)));
		mid.b = min(mid.b, sqrt(pow(q.a-.4, 2.) + pow(q.c-.2, 2.)));
		mid.c = min(mid.c, sqrt(pow(q.a-.4, 2.) + pow(q.b-.2, 2.)));
		// mid = (k*mid+q)/(k+1);
		if(abs(q) > 2.) break;
		q = q*q + c;
		// res = (k*res+getDist0(q))/(k+1);
		#endif
	}
	mid.d=res;
	return mid;
	return res;
}
double getDist(const Quaternion &q, double time) {
	// return getDist0(q);
	// return max(getDist0(q), q.c);
	const double t0 = M_PI*min(1., pow(time-.75, 2)*pow(time+.5, 2)*64./9.), t1 = M_PI;
	const double d0 = -sin(t0)*q.b + cos(t0)*q.c;
	const double d1 =  sin(t1)*q.b - cos(t1)*q.c;
	return max(getDist0(q), min(d0, d1));
}

const double EPS_N = 5.e-5;
vec3 getNormal(const Quaternion &q, double time) {
	// const double d = getDist(q, time);
	// return vec3(getDist(q + EPS_N * basis[0], time) - d,
	// 			getDist(q + EPS_N * basis[1], time) - d,
	// 			getDist(q + EPS_N * basis[2], time) - d).normalize();
	return vec3(getDist(q + EPS_N * basis[0], time) - getDist(q - EPS_N * basis[0], time),
				getDist(q + EPS_N * basis[1], time) - getDist(q - EPS_N * basis[1], time),
				getDist(q + EPS_N * basis[2], time) - getDist(q - EPS_N * basis[2], time)).normalize();
	// Quaternion qn = q;
	// mat4 J = mat4::Id;
	// for(int k = 0; k < MAX; ++k) {
	// 	const Quaternion c(-0.2,0.8,0,0);
	// 	if(abs(qn) > 20.) break;
	// 	J = (qn.matLeft()+qn.matRight()) * J;
	// 	qn = qn*qn + c;
	// }
	// const Quaternion nq = J.T()*qn;
	// return vec3(dot(basis[0], nq), dot(basis[1], nq), dot(basis[2], nq)).normalize();
}

const int D = 200;
const double EPS_R = 8.e-4;
const double OUT_R = 6.;
double castRay(const Quaternion &p, const Quaternion &r, double time) {
	double t = 0.;
	for(int step = 0; step < D; ++step) {
		const Quaternion q = p + t * r;
		const double d = getDist(q, time);
		if(d < EPS_R) break;
		t += .9*d;
		if(t > OUT_R) return -1.;
	}
	return getDist(p + t * r, time) < 10.*EPS_R ? t : -1.;
}

int main() {
	const int N_IM = 8;
	const int W = 2000;
	const int H = 1500;
	unsigned char* im = new unsigned char[3*W*H];

	for(int ni = 0; ni < N_IM; ++ni) {
		const double time = double(ni)/double(N_IM-1);
		basis[0] = Quaternion(cos(time*M_PI_2), 0., 0., sin(time*M_PI_2));
		const double angle_lim = 1.1;
		const double angle_cam = (2.*time-1.) * angle_lim;
		#if FUN == 1
		const vec3 cam(-3.*abs(sin(angle_cam))+2.*time, -3.*cos(angle_cam), 2.);
		#else
		const vec3 cam(-1.5*abs(sin(angle_cam)), -1.5*cos(angle_cam), 1.1);
		// const vec3 cam(-.2, -.5, -.3);
		#endif
		const vec3 look_at(-0.3, 0., 0.2);
		// const vec3 look_at = cam + vec3(.6, .4, .5);
		const double fov = 70.;

		const vec3 up(0., 0., 1.);
		const vec3 cv = (look_at - cam).normalize();
		vec3 cu = cross(cv, up).normalize();
		vec3 cw = cross(cu, cv);
		cu *= tan(fov*M_PI/360.);
		cw *= tan(fov*M_PI/360.);
		
		#pragma omp parallel for
		for(int i = 0; i < W; ++i) for(int j = 0; j < H; ++j) {
			vec3 col;
			const double cx = ((2.*i)/(W-1)-1.) * double(W)/double(H);
			const double cy = 1.-(2.*j)/(H-1);
			const vec3 rd = (cv + cx*cu + cy*cw).normalize();
			const Quaternion cq = v2q(cam);
			const Quaternion rq = v2q(rd);
			// col = getDist(Quaternion(2*cx, 2*cy), time) * vec3(1., 1., 1.);
			const double t = castRay(cq, rq, time);
			if(t < 0.) {
				col = .6*vec3(0.4, 0.7, 1.) - .5*rd.z*vec3(1.,1.,1.);
				const double h =  min(1., exp(-.6*(rd.z+.95)));
				col = (1.-h)*col + h*vec3(.15, .18, .21);
			} else {
				const Quaternion q = cq + t * rq;
				const vec3 normal = getNormal(q, time);
				const Quaternion lim = getLim(q + EPS_R*v2q(normal));

				// const double R = sqrt(3./4.);
				// const vec3 col_mat(.04+(lim.a+.5)/4., .02+(lim.b+R)/3.6, .04+(lim.c+R)/3.4);
				// const vec3 col_mat = .1*(normal + vec3(dot(lim, basis[0]), dot(lim, basis[1]), dot(lim, basis[2])))+.2;
				// const vec3 col_mat = .04*normal+(.02+.25*lim.a)*vec3(1.,1.,1.);
				const vec3 col_mat = (1.15*pow(lim.d, 1.4)+.05)*(.08*normal+.04*vec3(1.,1.,1.)+.15*vec3(lim.a, lim.b, lim.c))+.015*vec3(1.,1.,1.);

				const vec3 sun_dir = vec3(10.*time-5., -4., 7.).normalize();
				const double sun_dif = clamp(dot(normal, sun_dir), 0., 1.);
				const double sun_sha = castRay(q + 2.*EPS_R*v2q(normal), v2q(sun_dir), time) < 0. ? 1. : .3;
				col += col_mat * vec3(4., 3.2, 2.4) * sun_dif * sun_sha;

				const vec3 sky_dir = up;
				const double sky_dif = clamp(.4 + .6*dot(normal, sky_dir), 0., 1.);
				col += col_mat * vec3(0.07, 0.38, .65) * sky_dif;

				col *= exp(-.03*t);
				// col = .5*normal+vec3(.5,.5,.5);
				// col = lim.a*vec3(.5,.5,.5);
				// col = vec3(lim.a, lim.b, lim.c)*.4+.8*vec3(.5,.5,.5);
			}
			im[3*(j*W+i)+0] = 254.9 * pow(clamp(col.x, 0., 1.), .5);
			im[3*(j*W+i)+1] = 254.9 * pow(clamp(col.y, 0., 1.), .5);
			im[3*(j*W+i)+2] = 254.9 * pow(clamp(col.z, 0., 1.), .5);
		}

		stbi_write_png(("haha" + to_string(ni	) + ".png").c_str(), W, H, 3, im, 0);
		cerr << ni << " done" << endl;
	}
	delete[] im;
	return 0;
}