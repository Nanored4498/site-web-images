#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace complex_literals;
typedef complex<double> cpx;

const int W = 2400;
const int H = 2400;

const double EPS = 1e-10;
const double FAR = 1e100;
const int    MAX = 280;

int K = 16;
cpx sub = .36;
double re0 = -0.3;
double re1 = 4.5;
double im0 = -3.;
double im1 = 3.;

const double gammaE = 0.5772156649015328606;
double p = 3.;
void compute(const cpx &z, cpx &F, cpx &dF) {
	F = pow(z, p)-1.;
	dF = p*pow(z, p-1.);
	// const cpx f = exp(-gammaE*z) / z;
	// const cpx df = - (gammaE + 1./z) * f;
	// cpx g = 1.;
	// cpx dg = 0.;
	// for(int k = 1; k <= K; ++k) {
	// 	const double ik = 1./k;
	// 	const cpx zk = z*ik;
	// 	const cpx d = 1. / (1. + zk);
	// 	g *= exp(zk) * d;
	// 	dg += ik * (1. - d);
	// }
	// F = f*g - sub;
	// dF = (df + f*dg)*g;
	// F = 0.;
	// dF = 0.;
	// cpx pz = 1.;
	// for(int k = 1; k <= K; ++k) {
	// 	dF += double(k) * pz;
	// 	pz *= z;
	// 	F += pz;
	// }
	// F -= sub;
}

double                 // OUTPUT the distance estimate
distance0
( cpx z0            // INPUT  starting point
, int nzero         // INPUT  number of zeros
, const cpx *zero   // INPUT  the zeros
, const cpx *zerop  // INPUT  the power of each zero
, int npole         // INPUT  number of poles
, const cpx *pole   // INPUT  the poles
, const cpx *polep  // INPUT  the power of each pole
, int *which        // OUTPUT the index of the zero converged to
) {
  cpx z = z0;
  cpx dz = 1.0;
  double eps = 0.001;    // root radius, should be as large as possible
  for (int k = 0; k < 1024; ++k) { // fixed iteration limit
    for (int i = 0; i < nzero; ++i) { // check if converged
      double e = abs(z - zero[i]);
      if (e < eps) {
        *which = i;
        return e * -log(e) / abs(dz); // compute distance
      }
    }
    dz *= (2./3.)*(1.-pow(z,-3.));
    z -= (z-pow(z,-2.))/3.;
  }
  *which = nzero;
  return -1;  // didn't converge
}

void func(int s) {
	unsigned char* im = new unsigned char[W*H*3];
	vector<double> count(W*H, 0);
	vector<cpx> lim(W*H);
	vector<int> converged;

	#pragma omp parallel for
	for(int i = 0; i < H; ++i) for(int j = 0; j < W; ++j) {
		cpx z(re0 + (j+.5) * (re1 - re0) / W, im0 + (i+.5) * (im1 - im0) / H);
			// cpx zero[3] = {1., exp(cpx(0,1.)*2.*M_PI/3.), exp(cpx(0,1.)*4.*M_PI/3.)};
			// cpx zerop[3] = {1., 1., 1.};
			// int which;
			// const int p = j+W*i;
			// lim[p] = distance0(z, 3, zero, zerop, 0, nullptr, nullptr, &which);
			// #pragma omp critical
			// converged.push_back(p);
			// continue;
		cpx F, dF, dG = 1.;
		int k = 0;
		compute(z, F, dF);
		double norm2F = F.real()*F.real() + F.imag()*F.imag();
		double norm2 = max(norm2F, z.real()*z.real() + z.imag()*z.imag());
		while(k < MAX && norm2F > EPS && norm2 < FAR) {
			++ k;
			dG *= ((p-1.)/p)*(1.-pow(z,-p));
			z -= F / dF;
			compute(z, F, dF);
			norm2F = F.real()*F.real() + F.imag()*F.imag();
			norm2 = max(norm2F, z.real()*z.real() + z.imag()*z.imag());
		}
		if(k == MAX) {
			im[3*(j+W*i)+0] = 0;
			im[3*(j+W*i)+1] = 0;
			im[3*(j+W*i)+2] = 0;
		} else if(norm2 >= FAR || isnan(norm2)) {
			im[3*(j+W*i)+0] = .4*max(250-20*k, 0)+40;
			im[3*(j+W*i)+1] = .2*min(abs(250-20*k), 250)+40;
			im[3*(j+W*i)+2] = .4*clamp(20*k-250, 0, 250)+40;
		} else {
			const int p = j+W*i;
			count[p] = double(k) + norm2F/EPS;
			for(int i = 0; i < 3; ++i) {
				cpx r = exp(double(i)*2i*M_PI/3.);
				if(abs(z-r) < .001) lim[p] = cpx(z.imag(), - abs(z-r) * log(abs(z-r)) / abs(dG));
				// if(abs(z-r) < .001) lim[p] = cpx(- abs(z-r) * log(abs(z-r)) / abs(dG), z.imag());
			}
			#pragma omp critical
			converged.push_back(p);
		}
	}

	int nCon = converged.size();

	{ // OT for red component
	sort(converged.begin(), converged.end(), [&](int i, int j) { return count[i] > count[j]; });
	for(int i = 0; i < nCon; ++i)
		im[3*converged[i]] = 255 * pow(double(i)/double(nCon), .4545);
	}
	{ // OT for green component
	sort(converged.begin(), converged.end(), [&](int i, int j) { return lim[i].real() < lim[j].real(); });
	for(int i = 0; i < nCon; ++i)
		im[3*converged[i]+1] = 255 * pow(double(i)/double(nCon), .4545);
	}
	{ // OT for blue component
	sort(converged.begin(), converged.end(), [&](int i, int j) { return lim[i].imag() < lim[j].imag(); });
	for(int i = 0; i < nCon; ++i)
		im[3*converged[i]+2] = 255 * pow(double(i)/double(nCon), .4545);
	}

	stbi_write_png(("haha" + to_string(s) + ".png").c_str(), W, H, 3, im, 0);
	delete[] im;
}

int main() {
			cpx zero[3] = {1., exp(cpx(0,1.)*2.*M_PI/3.), exp(cpx(0,1.)*4.*M_PI/3.)};
			cpx zerop[3] = {1., 1., 1.};
			int which;
			cout << distance0(2.-5i, 3, zero, zerop, 0, nullptr, nullptr, &which) << endl;
			return 0;
	double dx = 3.*.42;
	double dy = 3.*.42;
	const double ra = 1.73;
	const double ia = -2.78;
	const double rb = 0.*0.1;
	const double ib = 0.*0.8;
	for(int s = 0; s <= 35; ++s) {
		re0 = rb+.01*s - dx;
		re1 = rb+.01*s + dx;
		im0 = ib - dy;
		im1 = ib + dy;
		sub = 1.;
		func(s);
		p += .1;
		cerr << s << ' ' << sub << endl;
		break;
		sub *= 1.1037;
	}
	return 0;
}