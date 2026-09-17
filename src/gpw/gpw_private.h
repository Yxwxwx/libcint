#ifndef GPW_PRIVATE_H
#define GPW_PRIVATE_H
#include "gpw.h"
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef double complex Z;
#define GPW_PI 3.141592653589793238462643383279502884
#define GPW_MAXROOTS 10
#define GPW_MAXPOINTS 4096
typedef struct {
    int l,np,nc,nf;
    double center[3],sp;
    const double *exps,*coeffs;
    int powers[28][3];
    double c2s[28*13]; /* row-major (ncart,nsph); no s/p rescaling */
} Shell;

struct GPWWorkspace {
    size_t capacity;
    Z *a,*b;
    Z axis[3][7][13][7][10];
    uint64_t rules,special;
};

int gpw_rule(int n,Z t,Z *nodes,Z *weights,double *residual);
int gpw_nuclei_check(const int32_t *atoms,int32_t natm,const double *env,size_t nenv);
int gpw_one_electron(GPWWorkspace *work,const Shell *a,const double *k,int op,
                     const int32_t *atoms,int32_t natm,const double *env);
#endif
