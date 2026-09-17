#include "gpw_private.h"
#include "cint.h"

/* Validate every nucleus, including atoms with no basis functions. */
int gpw_nuclei_check(const int32_t *atoms,int32_t natm,const double *env,size_t nenv) {
    for (int i=0;i<natm;++i) {
        const int32_t *a=atoms+(size_t)i*ATM_SLOTS;
        int32_t coord=a[PTR_COORD],model=a[NUC_MOD_OF];
        if (coord<0 || (size_t)coord>nenv || nenv-(size_t)coord<3) return GPW_INPUT;
        for (int d=0;d<3;++d) if (!isfinite(env[coord+d])) return GPW_INPUT;
        if (model!=POINT_NUC && model!=GAUSSIAN_NUC && model!=FRAC_CHARGE_NUC)
            return GPW_RANGE;
        if (model==GAUSSIAN_NUC || model==FRAC_CHARGE_NUC) {
            int32_t ptr=a[model==GAUSSIAN_NUC?PTR_ZETA:PTR_FRAC_CHARGE];
            if (ptr<0 || (size_t)ptr>=nenv || !isfinite(env[ptr])) return GPW_INPUT;
            if (model==GAUSSIAN_NUC && env[ptr]<0) return GPW_INPUT;
        }
    }
    return 0;
}

static void moments(Z m[7],int l,Z shift,Z variance) {
    m[0]=1;
    if (l) m[1]=shift;
    for (int j=2;j<=l;++j) m[j]=shift*m[j-1]+(j-1)*variance*m[j-2];
}

int gpw_one_electron(GPWWorkspace *work,const Shell *a,const double *k,int op,
                     const int32_t *atoms,int32_t natm,const double *env) {
    double k2=0,angle=0;
    for (int d=0;d<3;++d) { k2+=k[d]*k[d]; angle+=k[d]*a->center[d]; }
    if (!isfinite(k2)||!isfinite(angle)) return GPW_RANGE;
    memset(work->a,0,(size_t)a->nf*a->nc*sizeof(Z));
    if (op==GPW_KIN && k2==0) return 0;
    for (int ip=0;ip<a->np;++ip) {
        const double p=a->exps[ip],x=k2/(4*p);
        Z shift[3],value[28]={0};
        for (int d=0;d<3;++d) shift[d]=I*(k[d]/p*.5);
        if (!isfinite(x)) return GPW_RANGE;
        if (op!=GPW_NUC) {
            Z m[3][7];
            for (int d=0;d<3;++d) moments(m[d],a->l,shift[d],.5/p);
            Z pref=pow(GPW_PI/p,1.5)*cexp(-x+I*angle)*a->sp;
            if (op==GPW_KIN) pref*=.5*k2;
            for (int j=0;j<a->nf;++j) {
                const int *l=a->powers[j];
                value[j]=pref*m[0][l[0]]*m[1][l[1]]*m[2][l[2]];
            }
        } else {
            const int n=a->l/2+1;
            for (int ia=0;ia<natm;++ia) {
                const int32_t *atom=atoms+(size_t)ia*ATM_SLOTS;
                double charge=atom[NUC_MOD_OF]==FRAC_CHARGE_NUC?
                    env[atom[PTR_FRAC_CHARGE]]:fabs((double)atom[CHARGE_OF]);
                if (charge==0) continue;
                double tau2=1;
                if (atom[NUC_MOD_OF]==GAUSSIAN_NUC && env[atom[PTR_ZETA]]>0) {
                    double zeta=env[atom[PTR_ZETA]];
                    tau2=zeta>p?1/(1+p/zeta):(zeta/p)/(1+zeta/p);
                }
                Z D[3]; double r2=0,kr=0;
                for (int d=0;d<3;++d) {
                    double r=a->center[d]-env[atom[PTR_COORD]+d];
                    D[d]=r+shift[d]; r2+=r*r; kr+=k[d]*r;
                }
                double radial=p*tau2*r2;
                Z t=(radial-tau2*x)+I*(tau2*kr),s[GPW_MAXROOTS],w[GPW_MAXROOTS];
                double residual;
                int status=gpw_rule(n,t,s,w,&residual);
                work->rules++;
                if (fabs(creal(t))>20||fabs(cimag(t))>12) work->special++;
                if (status) return status;
                /* w integrates exp(min(Re T,0)-T*u^2). Combine the scaling
                 * analytically to retain the nuclear tail when exp(-x) underflows. */
                double attenuation=creal(t)<0?-radial-(1-tau2)*x:-x;
                Z pref=-charge*(2*GPW_PI/p)*sqrt(tau2)*a->sp*cexp(attenuation+I*angle);
                for (int root=0;root<n;++root) {
                    Z u=tau2*s[root],m[3][7];
                    for (int d=0;d<3;++d)
                        moments(m[d],a->l,shift[d]-u*D[d],(1-u)*(.5/p));
                    for (int j=0;j<a->nf;++j) {
                        const int *l=a->powers[j];
                        value[j]+=pref*w[root]*m[0][l[0]]*m[1][l[1]]*m[2][l[2]];
                    }
                }
            }
        }
        for (int j=0;j<a->nf;++j) {
            if (!isfinite(creal(value[j]))||!isfinite(cimag(value[j]))) return GPW_NUMERIC;
            for (int c=0;c<a->nc;++c)
                work->a[(size_t)c*a->nf+j]+=a->coeffs[(size_t)c*a->np+ip]*value[j];
        }
    }
    return 0;
}
