#include "gpw_private.h"
#include "cint.h"
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef char gpw_requires_int32_fint[(sizeof(FINT)==sizeof(int32_t))?1:-1];

struct GPWOpt {
    int32_t natm,nbas;
    size_t nenv;
    int32_t *atoms,*basis;
    double *env;
    Shell *shells;
};

static int multiply(size_t a,size_t b,size_t *result) {
    if (b && a>SIZE_MAX/b) return GPW_RANGE;
    *result=a*b;
    return *result>PTRDIFF_MAX/sizeof(Z)?GPW_RANGE:0;
}

static int inenv(int32_t start,size_t n,size_t nenv) {
    return start>=0 && (size_t)start<=nenv && n<=nenv-(size_t)start;
}

static int overlaps(const void *p,size_t n,const void *q,size_t m) {
    if (!n||!m) return 0;
    uintptr_t a=(uintptr_t)p,b=(uintptr_t)q;
    return a<=b?b-a<n:a-b<m;
}

static int read_shell(Shell *s,int32_t shell,const int32_t *atoms,int32_t natm,
                      const int32_t *basis,int32_t nbas,const double *env,size_t nenv) {
    if (shell<0||shell>=nbas) return GPW_INPUT;
    const int32_t *b=basis+8*(size_t)shell;
    if (b[0]<0||b[0]>=natm||b[1]<0||b[1]>6||b[2]<1||b[3]<1||b[4]!=0) return GPW_RANGE;
    int32_t coord=atoms[6*(size_t)b[0]+1];
    size_t coeffs;
    if (multiply(b[2],b[3],&coeffs)||!inenv(coord,3,nenv)||
        !inenv(b[5],b[2],nenv)||!inenv(b[6],coeffs,nenv)) return GPW_INPUT;
    s->l=b[1]; s->np=b[2]; s->nc=b[3]; s->nf=(s->l+1)*(s->l+2)/2;
    s->exps=env+b[5]; s->coeffs=env+b[6];
    s->sp=s->l==0?.282094791773878143:s->l==1?.488602511902919921:1.;
    for (int d=0;d<3;++d) { s->center[d]=env[coord+d]; if (!isfinite(s->center[d])) return GPW_INPUT; }
    for (int p=0;p<s->np;++p) if (!(s->exps[p]>0)||!isfinite(s->exps[p])) return GPW_INPUT;
    for (size_t c=0;c<coeffs;++c) if (!isfinite(s->coeffs[c])) return GPW_INPUT;
    int i=0;
    for (int x=s->l;x>=0;--x) for (int y=s->l-x;y>=0;--y) {
        s->powers[i][0]=x; s->powers[i][1]=y; s->powers[i++][2]=s->l-x-y;
    }
    double identity[28*28]={0},spherical[28*13];
    for (int k=0;k<s->nf;++k) identity[k*s->nf+k]=1;
    double *matrix=CINTc2s_ket_sph(spherical,s->nf,identity,s->l);
    for (int c=0;c<s->nf;++c) for (int h=0;h<2*s->l+1;++h)
        s->c2s[c*(2*s->l+1)+h]=matrix[h*s->nf+c];
    return GPW_SUCCESS;
}

static int environment(const int32_t *atoms,int32_t natm,const int32_t *basis,
                        int32_t nbas,const double *env,size_t nenv) {
    if (!atoms||!basis||!env||natm<0||nbas<0||nenv<20) return GPW_INPUT;
    if (env[PTR_RANGE_OMEGA]!=0.) return GPW_RANGE;
    size_t size;
    if (multiply(natm,ATM_SLOTS,&size)||multiply(nbas,BAS_SLOTS,&size)||
        multiply(nenv,sizeof(double),&size)) return GPW_RANGE;
    return GPW_SUCCESS;
}

int gpw_abi_version(void) { return 1; }
int gpw_integer_bytes(void) { return sizeof(FINT); }
const char *gpw_build_version(void) { return "gpw-0.2 libcint-" CINT_VERSION; }
const char *gpw_error_string(int code) {
    switch(code) {
    case 0: return "success";
    case 1: return "screened to zero";
    case GPW_INPUT: return "invalid input, capacity, pointer or env offset";
    case GPW_RANGE: return "unsupported angular momentum, operator or numerical range";
    case GPW_NUMERIC: return "complex quadrature or integral failed numerical checks";
    case GPW_MEMORY: return "allocation failed";
    case GPW_WORKSPACE: return "workspace capacity too small";
    case GPW_ABI: return "incompatible ABI";
    case GPW_STALE_OPT: return "molecule data changed after opt creation";
    default: return "unknown GPW status";
    }
}

void gpw_opt_destroy(GPWOpt *p) {
    if (p) { free(p->atoms); free(p->basis); free(p->env); free(p->shells); free(p); }
}

int gpw_opt_create(GPWOpt **out,const int32_t *atoms,int32_t natm,const int32_t *basis,
                    int32_t nbas,const double *env,size_t nenv) {
    if (!out) return GPW_INPUT;
    *out=NULL;
    int status=environment(atoms,natm,basis,nbas,env,nenv);
    if (status) return status;
    GPWOpt *p=calloc(1,sizeof(*p));
    if (!p) return GPW_MEMORY;
    p->natm=natm; p->nbas=nbas; p->nenv=nenv;
    p->atoms=malloc((size_t)natm*6*sizeof(int32_t));
    p->basis=malloc((size_t)nbas*8*sizeof(int32_t));
    p->env=malloc(nenv*sizeof(double));
    p->shells=calloc(nbas,sizeof(Shell));
    if ((!p->atoms&&natm)||(!p->basis&&nbas)||!p->env||(!p->shells&&nbas)) {
        gpw_opt_destroy(p); return GPW_MEMORY;
    }
    memcpy(p->atoms,atoms,(size_t)natm*6*sizeof(int32_t));
    memcpy(p->basis,basis,(size_t)nbas*8*sizeof(int32_t));
    memcpy(p->env,env,nenv*sizeof(double));
    for (int i=0;i<nbas;++i) {
        status=read_shell(p->shells+i,i,p->atoms,natm,p->basis,nbas,p->env,nenv);
        if (status) { gpw_opt_destroy(p); return status; }
    }
    *out=p;
    return 0;
}

int gpw_workspace_create(GPWWorkspace **out,size_t capacity) {
    if (!out) return GPW_INPUT;
    *out=NULL;
    size_t size;
    if (multiply(capacity,3*sizeof(Z),&size)) return GPW_RANGE;
    GPWWorkspace *w=malloc(sizeof(*w));
    if (!w) return GPW_MEMORY;
    w->a=malloc(size?size:sizeof(Z));
    if (!w->a) { free(w); return GPW_MEMORY; }
    w->b=w->a+capacity; w->c=w->b+capacity; w->capacity=capacity;
    w->params=NULL; w->param_capacity=0;
    w->tile=NULL; w->tile_capacity=0; w->tile_lane=-1;
    w->rules=0; w->special=0;
    *out=w;
    return 0;
}
void gpw_workspace_destroy(GPWWorkspace *w) { if (w) { free(w->tile); free(w->params); free(w->a); free(w); } }
size_t gpw_workspace_bytes(const GPWWorkspace *w) {
    return w?sizeof(*w)+w->capacity*3*sizeof(Z)+w->param_capacity*sizeof(PrimitiveData)+
        w->tile_capacity*sizeof(GPWLowRule):0;
}
uint64_t gpw_workspace_rules(const GPWWorkspace *w) { return w?w->rules:0; }
uint64_t gpw_workspace_special(const GPWWorkspace *w) { return w?w->special:0; }

/* The real Gaussian product and exponent factors are independent of the
 * wavevectors. Prepare once for the entire shell batch, not once per k. */
static int prepare_primitives(GPWWorkspace *work,const Shell *a,const Shell *b,const Shell *c) {
    size_t count,bytes;
    if (multiply(a->np,b->np,&count)||multiply(count,c?c->np:1,&count)||
        multiply(count,sizeof(PrimitiveData),&bytes)) return GPW_RANGE;
    if (count>work->param_capacity) {
        PrimitiveData *params=realloc(work->params,bytes);
        if (!params) return GPW_MEMORY;
        work->params=params; work->param_capacity=count;
    }
    size_t index=0;
    for (int ic=0;ic<(c?c->np:1);++ic) for (int ib=0;ib<b->np;++ib) {
        double beta=b->exps[ib],gamma=c?c->exps[ic]:0.,q=beta+gamma;
        double center[3],phase=0;
        for (int d=0;d<3;++d) {
            center[d]=c?(beta*b->center[d]+gamma*c->center[d])/q:b->center[d];
            if (c) phase-=beta*gamma/q*pow(b->center[d]-c->center[d],2);
        }
        for (int ia=0;ia<a->np;++ia) {
            double p=a->exps[ia];
            PrimitiveData *v=work->params+index++;
            v->inv2p=.5/p; v->inv2q=.5/q; v->rho=p*q/(p+q);
            v->pfrac=p/(p+q); v->qfrac=q/(p+q); v->cross=.5/(p+q);
            v->pref=2*pow(GPW_PI,2.5)/(p*q*sqrt(p+q))*a->sp*b->sp*(c?c->sp:1.);
            v->phase=phase;
            memcpy(v->center,center,sizeof(center));
        }
    }
    return 0;
}

static inline void axis_recurrence(Z g[7][13][7][10],int root,int la,int lb,int lc,
                                  Z base,Z da,Z db,Z va,Z vb,Z cross,double bc) {
#define G(i,j,h) g[i][j][h][root]
    G(0,0,0)=base;
    for (int j=1;j<=lb+lc;++j)
        G(0,j,0)=db*G(0,j-1,0)+(j>1?(j-1)*vb*G(0,j-2,0):0.);
    for (int i=1;i<=la;++i) for (int j=0;j<=lb+lc;++j)
        G(i,j,0)=da*G(i-1,j,0)+(i>1?(i-1)*va*G(i-2,j,0):0.)+
                                   (j?j*cross*G(i-1,j-1,0):0.);
    for (int h=1;h<=lc;++h) for (int i=0;i<=la;++i)
        for (int j=0;j<=lb+lc-h;++j)
            G(i,j,h)=G(i,j+1,h-1)+bc*G(i,j,h-1);
#undef G
}

static void primitive_centers(const Shell *a,const Shell *c,const PrimitiveData *v,
                              const double *k,const double *kp,Z P[3],Z Q[3],Z D[3]) {
    for (int d=0;d<3;++d) {
        P[d]=a->center[d]+I*k[d]*v->inv2p;
        Q[d]=v->center[d];
        if (!c) Q[d]-=I*kp[d]*v->inv2q;
        D[d]=P[d]-Q[d];
    }
}

static Z primitive_argument(const Shell *a,const Shell *b,const Shell *c,
                            const PrimitiveData *v,const double *k,const double *kp,
                            Z P[3],Z Q[3],Z D[3],Z *phase) {
    Z t=0;
    *phase=v->phase;
    primitive_centers(a,c,v,k,kp,P,Q,D);
    for (int d=0;d<3;++d) {
        *phase+=I*k[d]*a->center[d]-k[d]*k[d]*(.5*v->inv2p);
        if (!c)
            *phase-=I*kp[d]*b->center[d]+kp[d]*kp[d]*(.5*v->inv2q);
        t+=v->rho*D[d]*D[d];
    }
    return t;
}

static int primitive(GPWWorkspace *work,const Shell *a,const Shell *b,const Shell *c,
                     const PrimitiveData *v,const double *k,const double *kp,double coeff) {
    const int lc=c?c->l:0, n=(a->l+b->l+lc)/2+1;
    Z P[3],Q[3],D[3],phase,pref;
    Z local_s[10],local_w[10]; const Z *s=local_s,*weights=local_w;
    if (work->tile_lane>=0) {
        const GPWLowRule *rule=work->tile+(v-work->params)*GPW_LANES+work->tile_lane;
        s=rule->nodes; weights=rule->weights;
        primitive_centers(a,c,v,k,kp,P,Q,D);
        pref=rule->pref*coeff;
    } else {
        Z t=primitive_argument(a,b,c,v,k,kp,P,Q,D,&phase);
        double residual;
        int status=gpw_rule(n,t,local_s,local_w,&residual);
        work->rules++;
        if (fabs(creal(t))>20||fabs(cimag(t))>12) work->special++;
        if (status) return status;
        pref=v->pref*cexp(phase-fmin(creal(t),0.))*coeff;
    }
    for (int root=0;root<n;++root) {
        Z va=(1-v->qfrac*s[root])*v->inv2p, vb=(1-v->pfrac*s[root])*v->inv2q;
        Z cross=s[root]*v->cross;
        for (int d=0;d<3;++d) {
            Z da=P[d]-a->center[d]-v->qfrac*s[root]*D[d];
            Z db=Q[d]-b->center[d]+v->pfrac*s[root]*D[d];
            Z base=d==2?weights[root]*pref:1.;
            double bc=c?b->center[d]-c->center[d]:0.;
            /* Fixed low-l bounds let the compiler expand the same recurrence,
             * matching libcint's one/two-root specialization without new formulas. */
#define AXIS(A,B,C) axis_recurrence(work->axis[d],root,A,B,C,base,da,db,va,vb,cross,bc)
#define LOW(A,B,C) case (A*16+B*4+C): AXIS(A,B,C); break
            if (n<=2) switch (a->l*16+b->l*4+lc) {
                LOW(0,0,0); LOW(1,0,0); LOW(0,1,0); LOW(0,0,1);
                LOW(2,0,0); LOW(0,2,0); LOW(0,0,2);
                LOW(1,1,0); LOW(1,0,1); LOW(0,1,1);
                LOW(3,0,0); LOW(0,3,0); LOW(0,0,3);
                LOW(2,1,0); LOW(2,0,1); LOW(1,2,0);
                LOW(0,2,1); LOW(1,0,2); LOW(0,1,2); LOW(1,1,1);
            } else { AXIS(a->l,b->l,lc); }
#undef LOW
#undef AXIS
        }
    }
    return 0;
}

static int prepare_rules(GPWWorkspace *work,const Shell *a,const Shell *b,const Shell *c,
                         const double *k,const double *kp,int count) {
    size_t size,bytes;
    if (multiply(a->np,b->np,&size)||multiply(size,c?c->np:1,&size)||
        multiply(size,GPW_LANES,&size)||multiply(size,sizeof(GPWLowRule),&bytes)) return GPW_RANGE;
    if (size>work->tile_capacity) {
        GPWLowRule *tile=realloc(work->tile,bytes);
        if (!tile) return GPW_MEMORY;
        work->tile=tile; work->tile_capacity=size;
    }
    const int n=(a->l+b->l+(c?c->l:0))/2+1;
    Z unit_phase[GPW_LANES];
    for (int lane=0;lane<count;++lane) {
        double angle=0.;
        for (int d=0;d<3;++d) {
            angle+=k[3*lane+d]*a->center[d];
            if (!c) angle-=kp[3*lane+d]*b->center[d];
        }
        unit_phase[lane]=cos(angle)+I*sin(angle);
    }
    for (int ic=0;ic<(c?c->np:1);++ic) {
        if (c && c->nc==1 && c->coeffs[ic]==0.) continue;
        for (int ib=0;ib<b->np;++ib) {
            if (b->nc==1 && b->coeffs[ib]==0.) continue;
            for (int ia=0;ia<a->np;++ia) {
                if (a->nc==1 && a->coeffs[ia]==0.) continue;
                size_t index=((size_t)ic*b->np+ib)*a->np+ia;
                Z t[GPW_LANES];
                for (int lane=0;lane<count;++lane) {
                    Z P[3],Q[3],D[3],phase;
                    t[lane]=primitive_argument(a,b,c,work->params+index,k+3*lane,
                                               kp?kp+3*lane:NULL,P,Q,D,&phase);
                    work->tile[index*GPW_LANES+lane].pref=
                        work->params[index].pref*(exp(creal(phase)-fmin(creal(t[lane]),0.))*unit_phase[lane]);
                    work->rules++;
                    if (fabs(creal(t[lane]))>20||fabs(cimag(t[lane]))>12) work->special++;
                }
                int status=gpw_low_rules(n,t,count,work->tile+index*GPW_LANES);
                if (status) return status;
            }
        }
    }
    return 0;
}

static int shell_integral(GPWWorkspace *work,const Shell *a,const Shell *b,const Shell *c,
                         const double *k,const double *kp) {
    const int n=(a->l+b->l+(c?c->l:0))/2+1, fc=c?c->nf:1;
    const int cc_count=c?c->nc:1;
    const size_t na=(size_t)a->nf*a->nc, nb=(size_t)b->nf*b->nc;
    size_t block=na*nb*cc_count*fc;
    memset(work->a,0,block*sizeof(Z));
    /* Contract one shell at a time, as in CINT2e_loop. Single contractions
     * alias the next stage and put their coefficients in the primitive prefactor. */
    Z *ab=cc_count==1?work->a:work->c;
    Z *av=b->nc==1?ab:work->b;
    for (int ic=0;ic<(c?c->np:1);++ic) {
        double cf=c&&cc_count==1?c->coeffs[ic]:1.;
        if (cf==0.) continue;
        if (cc_count>1) memset(ab,0,na*nb*fc*sizeof(Z));
        for (int ib=0;ib<b->np;++ib) {
            double bf=b->nc==1?b->coeffs[ib]:1.;
            if (bf==0.) continue;
            if (b->nc>1) memset(av,0,na*b->nf*fc*sizeof(Z));
            for (int ia=0;ia<a->np;++ia) {
                double af=a->nc==1?a->coeffs[ia]:1.;
                if (af==0.) continue;
                const PrimitiveData *params=work->params+((size_t)ic*b->np+ib)*a->np+ia;
                int status=primitive(work,a,b,c,params,k,kp,af*bf*cf);
                if (status) return status;
                for (int i=0;i<a->nf;++i) for (int j=0;j<b->nf;++j) for (int h=0;h<fc;++h) {
                    const int zero[3]={0};
                    const int *ap=a->powers[i], *bp=b->powers[j], *cp=c?c->powers[h]:zero;
                    const Z *x=work->axis[0][ap[0]][bp[0]][cp[0]],
                            *y=work->axis[1][ap[1]][bp[1]][cp[1]],
                            *z=work->axis[2][ap[2]][bp[2]][cp[2]];
                    Z v=x[0]*y[0]*z[0];
                    if (n==2) v+=x[1]*y[1]*z[1];
                    else if (n>2) for (int root=1;root<n;++root) v+=x[root]*y[root]*z[root];
                    if (!isfinite(creal(v))||!isfinite(cimag(v))) return GPW_NUMERIC;
                    for (int ca=0;ca<a->nc;++ca) {
                        double coeff=a->nc==1?1.:a->coeffs[(size_t)ca*a->np+ia];
                        if (coeff!=0.) av[(((size_t)ca*a->nf+i)*b->nf+j)*fc+h]+=coeff*v;
                    }
                }
            }
            if (b->nc>1) for (size_t ai=0;ai<na;++ai) for (int cb=0;cb<b->nc;++cb) {
                double coeff=b->coeffs[(size_t)cb*b->np+ib];
                if (coeff==0.) continue;
                Z *dst=ab+(ai*nb+cb*b->nf)*fc;
                const Z *src=av+ai*b->nf*fc;
                for (int j=0;j<b->nf*fc;++j) dst[j]+=coeff*src[j];
            }
        }
        if (cc_count>1) for (size_t ij=0;ij<na*nb;++ij) for (int cc=0;cc<cc_count;++cc) {
            double coeff=c->coeffs[(size_t)cc*c->np+ic];
            if (coeff==0.) continue;
            for (int h=0;h<fc;++h) work->a[(ij*cc_count+cc)*fc+h]+=coeff*ab[ij*fc+h];
        }
    }
    return 0;
}

static Z *spherical(GPWWorkspace *work,const Shell **ss,int rank,size_t *dims,int ncomp) {
    Z *src=work->a,*dst=work->b;
    for (int axis=0;axis<rank;++axis) {
        const Shell *s=ss[axis];
        if (s->l<2) continue;
        size_t outer=ncomp,inner=1;
        for (int i=0;i<axis;++i) outer*=dims[i];
        for (int i=axis+1;i<rank;++i) inner*=dims[i];
        const int ns=2*s->l+1;
        const size_t newdim=(size_t)ns*s->nc,old=dims[axis];
        memset(dst,0,outer*newdim*inner*sizeof(Z));
        for (size_t o=0;o<outer;++o) for (int ctr=0;ctr<s->nc;++ctr)
            for (int h=0;h<ns;++h) for (int c=0;c<s->nf;++c) {
                double coeff=s->c2s[c*ns+h];
                if (coeff==0.) continue;
                const Z *from=src+(o*old+(size_t)ctr*s->nf+c)*inner;
                Z *to=dst+(o*newdim+(size_t)ctr*ns+h)*inner;
                for (size_t j=0;j<inner;++j) to[j]+=coeff*from[j];
            }
        dims[axis]=newdim;
        Z *tmp=src; src=dst; dst=tmp;
    }
    return src;
}

static int integral_batch(double *out,size_t capacity,int32_t kind,int32_t op,int32_t cart,
    const int32_t *shls,const double *k,const double *kp,size_t count,
    const int32_t *atoms,int32_t natm,const int32_t *basis,int32_t nbas,
    const double *env,size_t nenv,const GPWOpt *opt,GPWWorkspace *work) {
    int status=environment(atoms,natm,basis,nbas,env,nenv);
    if (status) return status;
    if ((kind<1||kind>3)||(kind==1&&(op<GPW_OVLP||op>GPW_DIPOLE))||
        (cart!=0&&cart!=1)||!shls) return GPW_INPUT;
    const int ncomp=kind==1&&op==GPW_DIPOLE?3:1;
    if (ncomp==3) for (int d=0;d<3;++d)
        if (!isfinite(env[PTR_COMMON_ORIG+d])) return GPW_INPUT;
    if (kind==1 && op==GPW_NUC) {
        status=gpw_nuclei_check(atoms,natm,env,nenv);
        if (status) return status;
    }
    size_t nvec;
    if (multiply(count,3,&nvec)) return GPW_RANGE;
    if (count && (!out||!k||(kind==2&&!kp))) return GPW_INPUT;
    for (size_t i=0;i<nvec;++i) if (!isfinite(k[i])||(kind==2&&!isfinite(kp[i]))) return GPW_INPUT;
    if (opt && (opt->natm!=natm||opt->nbas!=nbas||opt->nenv!=nenv||
        memcmp(opt->atoms,atoms,(size_t)natm*6*sizeof(int32_t))||
        memcmp(opt->basis,basis,(size_t)nbas*8*sizeof(int32_t))||memcmp(opt->env,env,nenv*sizeof(double))))
        return GPW_STALE_OPT;
    Shell local[3]; const Shell *ss[3];
    size_t block=ncomp,cartblock=ncomp;
    for (int i=0;i<kind;++i) {
        if (shls[i]<0||shls[i]>=nbas) return GPW_INPUT;
        if (opt) ss[i]=opt->shells+shls[i];
        else {
            status=read_shell(local+i,shls[i],atoms,natm,basis,nbas,env,nenv);
            if (status) return status;
            ss[i]=local+i;
        }
        if (multiply(cartblock,(size_t)ss[i]->nf*ss[i]->nc,&cartblock)||
            multiply(block,(size_t)(cart?ss[i]->nf:2*ss[i]->l+1)*ss[i]->nc,&block)) return GPW_RANGE;
    }
    size_t required;
    if (multiply(block,count,&required)||capacity>SIZE_MAX/(2*sizeof(double))) return GPW_RANGE;
    if (capacity<required) return GPW_INPUT;
    if (!count) return GPW_SUCCESS;
    const size_t outbytes=required*2*sizeof(double);
    if (overlaps(out,outbytes,env,nenv*sizeof(double)) ||
        overlaps(out,outbytes,atoms,(size_t)natm*6*sizeof(int32_t)) ||
        overlaps(out,outbytes,basis,(size_t)nbas*8*sizeof(int32_t)) ||
        overlaps(out,outbytes,shls,(size_t)kind*sizeof(int32_t)) ||
        overlaps(out,outbytes,k,nvec*sizeof(double)) ||
        (kind==2 && overlaps(out,outbytes,kp,nvec*sizeof(double)))) return GPW_INPUT;
    GPWWorkspace *owned=NULL;
    if (!work) {
        status=gpw_workspace_create(&owned,cartblock);
        if (status) return status;
        work=owned;
    }
    if (work->capacity<cartblock) { gpw_workspace_destroy(owned); return GPW_WORKSPACE; }
    if (kind>1) {
        status=prepare_primitives(work,ss[0],ss[1],kind==3?ss[2]:NULL);
        if (status) { memset(out,0,outbytes); gpw_workspace_destroy(owned); return status; }

    }
    uint64_t rules=0,special=0,old_rules=work->rules,old_special=work->special;
    int lanes=kind>1 && count>1 && ss[0]->l+ss[1]->l+(kind==3?ss[2]->l:0)<=3?GPW_LANES:1;
#ifdef _OPENMP
    int threads=omp_in_parallel()?1:omp_get_max_threads();
    size_t tiles=(count-1)/lanes+1;
    if (tiles<(size_t)threads) threads=(int)tiles;
#pragma omp parallel if(threads>1) num_threads(threads) reduction(min:status) reduction(+:rules,special)
#endif
    {
        GPWWorkspace *scratch=work;
        int local_status=0;
#ifdef _OPENMP
        if (omp_get_thread_num()!=0) {
            local_status=gpw_workspace_create(&scratch,cartblock);
            /* Only the prepared primitive table is shared; all scratch is private. */
            if (!local_status) scratch->params=work->params;
        }
#endif
        uint64_t before_rules=scratch?scratch->rules:0,before_special=scratch?scratch->special:0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (size_t first=0;first<count;first+=lanes) {
            if (local_status) continue;
            int used=count-first<(size_t)lanes?(int)(count-first):lanes;
            if (lanes>1) {
                local_status=prepare_rules(scratch,ss[0],ss[1],kind==3?ss[2]:NULL,
                                           k+3*first,kind==2?kp+3*first:NULL,used);
                if (local_status) continue;
            }
            for (int lane=0;lane<used;++lane) {
                size_t ik=first+lane;
                scratch->tile_lane=lanes>1?lane:-1;
                local_status=kind==1?gpw_one_electron(scratch,ss[0],k+3*ik,op,atoms,natm,env):
                    shell_integral(scratch,ss[0],ss[1],kind==3?ss[2]:NULL,k+3*ik,kind==2?kp+3*ik:NULL);
                if (local_status) break;
                size_t dims[3];
                for (int i=0;i<kind;++i) dims[i]=(size_t)ss[i]->nf*ss[i]->nc;
                Z *value=cart?scratch->a:spherical(scratch,ss,kind,dims,ncomp);
                for (size_t i=0;i<block;++i) {
                    if (!isfinite(creal(value[i]))||!isfinite(cimag(value[i]))) { local_status=GPW_NUMERIC; break; }
                    out[2*(ik*block+i)]=creal(value[i]); out[2*(ik*block+i)+1]=cimag(value[i]);
                }
                if (local_status) break;
            }
            scratch->tile_lane=-1;
        }
        if (local_status<status) status=local_status;
        if (scratch) {
            rules+=scratch->rules-before_rules; special+=scratch->special-before_special;
            if (scratch!=work) {
                scratch->params=NULL;
                gpw_workspace_destroy(scratch);
            }
        }
    }
    work->rules=old_rules+rules; work->special=old_special+special;
    if (status) memset(out,0,outbytes);
    gpw_workspace_destroy(owned);
    return status;
}

int gpw_int2e_batch(double *out,size_t capacity,int32_t kind,int32_t cart,
    const int32_t *shls,const double *k,const double *kp,size_t count,
    const int32_t *atoms,int32_t natm,const int32_t *basis,int32_t nbas,
    const double *env,size_t nenv,const GPWOpt *opt,GPWWorkspace *work) {
    if (kind!=GPW_GPPG && kind!=GPW_GPGG) return GPW_INPUT;
    return integral_batch(out,capacity,kind,0,cart,shls,k,kp,count,
                          atoms,natm,basis,nbas,env,nenv,opt,work);
}

int gpw_int1e_batch(double *out,size_t capacity,int32_t op,int32_t cart,
    const int32_t shls[1],const double *k,size_t count,
    const int32_t *atoms,int32_t natm,const int32_t *basis,int32_t nbas,
    const double *env,size_t nenv,const GPWOpt *opt,GPWWorkspace *work) {
    return integral_batch(out,capacity,1,op,cart,shls,k,NULL,count,
                          atoms,natm,basis,nbas,env,nenv,opt,work);
}

#define GP(name,op,cart) \
int name(double *out,size_t capacity,const int32_t shls[1],const double k[3], \
 const int32_t *atoms,int32_t natm,const int32_t *basis,int32_t nbas, \
 const double *env,size_t nenv,const GPWOpt *opt,GPWWorkspace *work) { \
 return gpw_int1e_batch(out,capacity,op,cart,shls,k,1,atoms,natm,basis,nbas,env,nenv,opt,work); }
GP(gpw_int1e_ovlp_gp_cart,GPW_OVLP,1)
GP(gpw_int1e_ovlp_gp_sph,GPW_OVLP,0)
GP(gpw_int1e_kin_gp_cart,GPW_KIN,1)
GP(gpw_int1e_kin_gp_sph,GPW_KIN,0)
GP(gpw_int1e_nuc_gp_cart,GPW_NUC,1)
GP(gpw_int1e_nuc_gp_sph,GPW_NUC,0)
GP(gpw_int1e_dipole_gp_cart,GPW_DIPOLE,1)
GP(gpw_int1e_dipole_gp_sph,GPW_DIPOLE,0)

#define GPGG(name,cart) \
int name(double *out,size_t capacity,const int32_t shls[3],const double k[3], \
 const int32_t *atoms,int32_t natm,const int32_t *basis,int32_t nbas, \
 const double *env,size_t nenv,const GPWOpt *opt,GPWWorkspace *work) { \
 return gpw_int2e_batch(out,capacity,3,cart,shls,k,NULL,1,atoms,natm,basis,nbas,env,nenv,opt,work); }
#define GPPG(name,cart) \
int name(double *out,size_t capacity,const int32_t shls[2],const double k[3],const double kp[3], \
 const int32_t *atoms,int32_t natm,const int32_t *basis,int32_t nbas, \
 const double *env,size_t nenv,const GPWOpt *opt,GPWWorkspace *work) { \
 return gpw_int2e_batch(out,capacity,2,cart,shls,k,kp,1,atoms,natm,basis,nbas,env,nenv,opt,work); }
GPGG(gpw_int2e_gpgg_cart,1)
GPGG(gpw_int2e_gpgg_sph,0)
GPPG(gpw_int2e_gppg_cart,1)
GPPG(gpw_int2e_gppg_sph,0)
