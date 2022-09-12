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
	Quaternion(double a, double b=0., double c=0., double d=0.): a(a), b(b), c(c), d(d) {}

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

	inline Quaternion& operator+=(const Quaternion &q) {
		a += q.a; b += q.b; c += q.c; d += q.d;
		return *this;
	}
	inline Quaternion& operator-=(const Quaternion &q) {
		a -= q.a; b -= q.b; c -= q.c; d -= q.d;
		return *this;
	}
	inline Quaternion& operator*=(const Quaternion &q) { return *this = *this * q; }

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
const mat4 unitLeft  = Quaternion(1., 1., 1., 1.).matLeft();
const mat4 unitRight = Quaternion(1., 1., 1., 1.).matRight();

inline Quaternion v2q(const vec3 &v) { return v.x*basis[0] + v.y*basis[1] + v.z*basis[2]; }

const double EPS = 1e-5;
const int    MAX = 100;
const Quaternion rs[3] { 1., Quaternion(cos(2.*M_PI/3.), sin(2.*M_PI/3.)), Quaternion(cos(4.*M_PI/3.), sin(4.*M_PI/3.)) };
double getDist0(Quaternion q) {
	// cerr << q << endl;
	mat4 dq = mat4::Id;
	// double dq2 = 1.;
	for(int k = 0; k < MAX; ++k) {
		// 	const double d = abs(q - 1.);
		// 	// if(d < EPS) return - .5 * d * log(d) / dq;
		// 	if(d < EPS) return .25 * (1. - pow(d, pow(2., 1-k))) * d * pow(2., k) / dq;
		// {
		// 	const double nu = sqrt(3./4.);
		// 	const double nq = nu / sqrt(q.b*q.b + q.c*q.c + q.d*q.d);
		// 	const Quaternion proj(-.5, nq*q.b, nq*q.c, nq*q.d);
		// 	const double d = abs(q - proj);
		// 	// if(d < EPS) return - .5 * d * log(d) / dq;
		// 	if(d < EPS) return .25 * (1. - pow(d, pow(2., 1-k))) * d * pow(2., k) / dq;
		// }
		// const Quaternion iq = inv(q);
		// const Quaternion step = (1./3.) * (q - iq*iq);
		// dq *= 2. * abs(iq) * abs(step);
		// q -= step;
		Quaternion diff[3];
		for(int i = 0; i < 3; ++i) {
			diff[i] = q - rs[i];
			const double d = abs(diff[i]);
			if(d < EPS) {
				// cerr << (.25 * (1. - pow(d, pow(2., 1-k))) * d * pow(2., k) / dq) << ' ' << k << ' ' << d << ' ' << rs[i] << endl;
				// exit(0);
				// cerr << dq << ' ' << d << ' ' << dq.T()*diff[i] << ' ' << - .5 * log(d) * d*d / abs(dq.T()*diff[i]) << endl;
				// exit(0);
				// if(k > 80) cerr << k << ' ' << - log(d) * d*d / abs(dq.T()*diff[i]) << endl;
				// return - log(d) * d*d / abs(dq.T()*diff[i]);
				return -.5 * log(d) * (1 + log(d)*pow(2.,-k)) * d * (d/abs(dq.T()*diff[i]));
				// return .25 * (1. - pow(d, pow(2., 1-k))) * d * pow(2., k) / pow(det(dq), 1./4.);
			}
		}
		const Quaternion F = diff[0]*diff[1]*diff[2];
		const mat4 dF = (diff[1]*diff[2]).matRight()
						+ diff[0].matLeft()*diff[2].matRight()
						+ (diff[0]*diff[1]).matLeft();
		// const Quaternion h = Quaternion(.001, -.002, -.040, .003);
		// cerr << q << ' ' << F << ' ' << F+dF*h << ' ' << (q+h-rs[0])*(q+h-rs[1])*(q+h-rs[2]) << endl;
		// cerr << dF << endl;
		// cerr << 3*q*q << endl;
		// cerr << abs(3*q*q) << ' ' << pow(det(dF),.25) << endl;
		const mat4 ddF_F = (diff[1]*F + F*diff[2]).matRight()
							+ F.matLeft()*diff[2].matRight() + diff[0].matLeft()*F.matRight()
							+ (diff[0]*F + F*diff[1]).matLeft();
		const mat4 idF = inv(dF);
		// const mat4 dG = idF * ddF_F * idF;
		// cerr << dG << endl;
		// const Quaternion iq = inv(q);
		// cerr << 2./3 * iq * (q - iq*iq) << endl;
		// cerr << q << ' ' << q-step << ' ' << q-step+dG*h << ' ' << endl;
		// {
		// q += h;
		// Quaternion diff[3];
		// for(int i = 0; i < 3; ++i) diff[i] = q - rs[i];
		// const Quaternion F = diff[0]*diff[1]*diff[2];
		// const mat4 dF = (diff[1]*diff[2]).matRight()
		// 				+ diff[0].matLeft()*diff[2].matRight()
		// 				+ (diff[0]*diff[1]).matLeft();
		// const mat4 idF = inv(dF);
		// const Quaternion step = idF * F;
		// cerr << q-step << endl;
		// exit(0);
		// }
		// cerr << idF * ddF_F * idF << endl;
		const Quaternion step = idF*F;
		// cerr << dF*step << ' ' << F << ' ' << step*diff[1]*diff[2]+diff[0]*step*diff[2]+diff[0]*diff[1]*step << endl;
		dq = idF * ddF_F * idF * dq;
		// dq = dq * idF * ddF_F * idF;
		q -= idF * F;
	}
	cerr << q << endl;
	return 0.;
}
Quaternion getLim(Quaternion q) {
	for(int k = 0; k < MAX; ++k) {
		// if(abs(q - 1.) < EPS) return 1.;
		// const double nu = sqrt(3./4.);
		// const double nq = nu / sqrt(q.b*q.b + q.c*q.c + q.d*q.d);
		// const Quaternion proj(-.5, nq*q.b, nq*q.c, nq*q.d);
		// if(abs(q - proj) < EPS) return proj;
		// const Quaternion iq = inv(q);
		// q -= (1./3.) * (q - iq*iq);
		Quaternion diff[3];
		for(int i = 0; i < 3; ++i) {
			diff[i] = q - rs[i];
			if(abs(diff[i]) < EPS) return rs[i];
		}
		const Quaternion F = diff[0]*diff[1]*diff[2];
		const Quaternion dF = diff[1]*diff[2] + diff[0]*diff[2] + diff[0]*diff[1];
		const Quaternion idF = inv(dF);
		q -= idF * F;
	}
	return 0.;
}
double getDist(const Quaternion &q, double time) {
	// return getDist0(q);
	// return max(getDist0(q), q.c);
	const double t0 = M_PI*min(1., pow(time-.75, 2)*pow(time+.5, 2)*64./9.), t1 = M_PI;
	const double d0 = -sin(t0)*q.b + cos(t0)*q.c;
	const double d1 =  sin(t1)*q.b - cos(t1)*q.c;
	return max(getDist0(q), min(d0, d1));
}

const double EPS_N = 1e-4;
vec3 getNormal(const Quaternion &q, double time) {
	const double d = getDist(q, time);
	vec3 res = vec3(getDist(q + EPS_N * basis[0], time) - d,
				getDist(q + EPS_N * basis[1], time) - d,
				getDist(q + EPS_N * basis[2], time) - d).normalize();
	if(isnan(res.x)) cerr << res << ' '  << q << ' ' << d << ' ' << q + EPS_N * basis[0] << endl;
	assert(!isnan(res.x));
	return res;
}

const int D = 300;
const double EPS_R = 8.e-4;
const double OUT_R = 62.;
double castRay(const Quaternion &p, const Quaternion &r, double time) {
	double t = 0.;
	for(int step = 0; step < D; ++step) {
		const Quaternion q = p + t * r;
		const double d = getDist(q, time);
		if(d < EPS_R) break;
		t += .9*d;
		if(t > OUT_R) return -1.;
	}
	return t;
}

int main() {
	// {
	// Quaternion q(3., 1., -2., 1.), a(1., 1., -1., .0), b(0., 2., 1., -1.);
	// Quaternion h(.03, -.02, -.04, .01);
	// cerr << (q-a)*(q-b) << ' ' << (q+h-a)*(q+h-b) << ' ' << (q-a)*(q-b)+(q-b+q-a)*h << ' ' << (q-a)*(q-b)+h*(q-b)+(q-a)*h << endl;
	// cerr << (q-a)*(q-b) << ' ' << (q+h-a)*(q+h-b) << ' ' << (q-a)*(q-b)+((q-b).matLeft()+(q-a).matLeft())*h << ' ' << (q-a)*(q-b)+((q-b).matRight()+(q-a).matLeft())*h << endl;
	// cerr << q.matLeft()*inv(q.matLeft()) << endl;
	// cerr << abs(q) << ' ' << pow(det(q.matLeft()), 1./4.) << endl;
	// cerr << a.matLeft()*b.matLeft() - (a*b).matLeft() << endl;
	// cerr << b.matLeft()*a.matLeft() - (a*b).matLeft() << endl;
	// cerr << a.matRight()*b.matRight() - (a*b).matRight() << endl;
	// cerr << b.matRight()*a.matRight() - (a*b).matRight() << endl;
	// cerr << b.matRight()*a.matLeft() - a.matLeft()*b.matRight() << endl;
	// cerr << a.matRight()*b.matLeft() - b.matLeft()*a.matRight() << endl;
	// return 0.;
	// }
	// for(int i = 0; i < 3; ++i) {
	// 	const Quaternion r0(cos(2*i*M_PI/3.), sin(2*i*M_PI/3.));
	// 	Quaternion q = r0 + Quaternion(.00000000001, -.00000000002, -.00000000002, -.00000000002);
	// 	Quaternion diff[3];
	// 	for(int i = 0; i < 3; ++i) {
	// 		const Quaternion r(cos(2*i*M_PI/3.), sin(2*i*M_PI/3.));
	// 		diff[i%3] = q - r;
	// 	}
	// 	const Quaternion f = diff[0]*diff[1]*diff[2];
	// 	const Quaternion df = diff[1]*diff[2] + diff[0]*diff[2] + diff[0]*diff[1];
	// 	const Quaternion idf = inv(df);
	// 	const Quaternion F = q - idf*f;
	// 	cerr << log(abs(F-r0)) / log(abs(q-r0)) << endl;
	// 	cerr << abs(q-r0) / abs(F-r0) << endl;
	// }
	// 	cerr << 1. / abs((rs[1]-rs[0])*(rs[1]-rs[2])) << endl;
	// 	return 0;
	// const int N_IM = 80;
	// const int W = 2080;
	// const int H = 1560;
	const int N_IM = 8;
	const int W = 800;
	const int H = 600;
	unsigned char* im = new unsigned char[3*W*H];

	for(int ni = 0; ni < N_IM; ++ni) {
		const double time = double(ni)/double(N_IM-1);
		basis[0] = Quaternion(cos(time*M_PI_2), 0., 0., sin(time*M_PI_2));
		const double angle_lim = 1.1;
		const double angle_cam = (2.*time-1.) * angle_lim;
		const vec3 cam(-3.*abs(sin(angle_cam))+2.*time, -3.*cos(angle_cam), 2.);
		const vec3 look_at(0., 0., 0.);
		const double fov = 70.;

		const vec3 up(0., 0., 1.);
		const vec3 cv = (look_at - cam).normalize();
		vec3 cu = cross(cv, up).normalize();
		vec3 cw = cross(cu, cv);
		cu *= tan(fov*M_PI/360.);
		cw *= tan(fov*M_PI/360.);
		
		// #pragma omp parallel for
		for(int i = 0; i < W; ++i) for(int j = 0; j < H; ++j) {
			vec3 col;
			const double cx = ((2.*i)/(W-1)-1.) * double(W)/double(H);
			const double cy = 1.-(2.*j)/(H-1);
			const vec3 rd = (cv + cx*cu + cy*cw).normalize();
			const Quaternion cq = v2q(cam);
			const Quaternion rq = v2q(rd);
			// col = getDist(Quaternion(2*cx, 2*cy), time) * vec3(1., 1., 1.);
			// cerr << col << endl;
			const double t = castRay(cq, rq, time);
			if(t < 0.) {
				col = .6*vec3(0.4, 0.7, 1.) - .5*rd.z*vec3(1.,1.,1.);
				const double h =  min(1., exp(-.6*(rd.z+.95)));
				col = (1.-h)*col + h*vec3(.15, .18, .21);
			} else {
				const Quaternion q = cq + t * rq;
				const vec3 normal = getNormal(q, time);
				const Quaternion lim = getLim(q + EPS_R*v2q(normal));

				const double R = sqrt(3./4.);
				const vec3 col_mat(.04+(lim.a+.5)/4., .02+(lim.b+R)/3.6, .04+(lim.c+R)/3.4);

				const vec3 sun_dir = vec3(10.*time-5., -4., 7.).normalize();
				const double sun_dif = clamp(dot(normal, sun_dir), 0., 1.);
				const double sun_sha = castRay(q + 2.*EPS_R*v2q(normal), v2q(sun_dir), time) < 0. ? 1. : .3;
				col += col_mat * vec3(4., 3.2, 2.4) * sun_dif * sun_sha;

				const vec3 sky_dir = up;
				const double sky_dif = clamp(.4 + .6*dot(normal, sky_dir), 0., 1.);
				col += col_mat * vec3(0.07, 0.38, .65) * sky_dif;

				col *= exp(-.03*t);
			}
			im[3*(j*W+i)+0] = 254.9 * pow(clamp(col.x, 0., 1.), .5);
			im[3*(j*W+i)+1] = 254.9 * pow(clamp(col.y, 0., 1.), .5);
			im[3*(j*W+i)+2] = 254.9 * pow(clamp(col.z, 0., 1.), .5);
		}

		stbi_write_png(("haha" + to_string(ni) + ".png").c_str(), W, H, 3, im, 0);
		cerr << ni << " done" << endl;
	}
	delete[] im;
	return 0;
}