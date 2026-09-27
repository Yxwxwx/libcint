/* Independent discrete Stieltjes construction in a scaled node variable.
 * Orthogonal-polynomial products are bilinear, never conjugated. A standard
 * unitary QR eigenvalue calculation is separate from that inner product.
 * See GPW_README.md for the conventions and numerical limits. */
#include "gpw_private.h"
#include "gpw_legendre.h"

typedef struct { int n; double y[GPW_MAXPOINTS]; Z w[GPW_MAXPOINTS];
                 double offset, factor; } Measure;

/* Extra precision only inside the complex rule construction; the ABI stays double. */
typedef long double complex LZ;

static int measure(Z t,Measure *m) {
    double a=creal(t), b=cimag(t), scale=fmin(a,0.);
    if (!isfinite(a)||!isfinite(b)) return GPW_INPUT;
    m->n=0; m->offset=0; m->factor=1;
    double ends[130]; int count=0;
    ends[count++]=0.;
    if (fabs(a)>20.) {
        double ab=fabs(a);
        m->factor=1/ab;
        if (a<0) { m->offset=1; m->factor=-1/ab; }
        for (double x=1; x<ab && count<129; x*=2)
            ends[count++]=a>0?sqrt(x/ab):sqrt(1-x/ab);
        ends[count++]=1.;
        /* At negative a the t endpoints were generated in descending order. */
        for (int i=1;i<count;++i) for (int j=i;j>0 && ends[j]<ends[j-1];--j) {
            double v=ends[j]; ends[j]=ends[j-1]; ends[j-1]=v;
        }
    } else ends[count++]=1.;
    for (int panel=1;panel<count;++panel) {
        double left=ends[panel-1], right=ends[panel];
        double oscillation=fabs(b)*(right*right-left*left);
        if (!isfinite(oscillation)||oscillation>12.*(GPW_MAXPOINTS/32)) return GPW_RANGE;
        int split=(int)fmax(1.,ceil(oscillation/12.));
        if (split>(GPW_MAXPOINTS-m->n)/32) return GPW_RANGE;
        for (int part=0;part<split;++part) {
            double lo=left+(right-left)*part/split, width=(right-left)/split;
            for (int j=0;j<32;++j) {
                double v=lo+width*gpw_gl_x[j], s=v*v;
                int i=m->n++;
                m->y[i]=(s-m->offset)/m->factor;
                m->w[i]=width*gpw_gl_w[j]*cexp(scale-t*s);
            }
        }
    }
    return GPW_SUCCESS;
}

/* Dense shifted QR for a matrix of order <=10. Householder conjugation here
 * is the usual QR factorization, not the Stieltjes polynomial inner product. */
static int eigenvalues(int n,const LZ *alpha,const LZ *beta,LZ *roots) {
    if (n==1) { roots[0]=alpha[0]; return 0; }
    if (n==2) {
        LZ delta= csqrtl((alpha[0]-alpha[1])*(alpha[0]-alpha[1])+4*beta[1]);
        roots[0]=(alpha[0]+alpha[1]+delta)*.5;
        roots[1]=(alpha[0]+alpha[1]-delta)*.5;
        return 0;
    }
    if (n==3) {
        LZ c2=-alpha[0]-alpha[1]-alpha[2];
        LZ c1=alpha[0]*alpha[1]+alpha[2]*(alpha[0]+alpha[1])-beta[1]-beta[2];
        LZ c0=-alpha[0]*alpha[1]*alpha[2]+alpha[2]*beta[1]+alpha[0]*beta[2];
        LZ p=c1-c2*c2/3., q=2*c2*c2*c2/27.-c2*c1/3.+c0;
        LZ delta= csqrtl(q*q/4.+p*p*p/27.);
        LZ plus=-q/2.+delta,minus=-q/2.-delta;
        LZ u= cpowl( cabsl(plus)> cabsl(minus)?plus:minus,1.L/3.L);
        if ( cabsl(u)>DBL_MIN) {
            LZ v=-p/(3.*u),omega=-.5+I*.86602540378443864676L;
            roots[0]=u+v-c2/3.;
            roots[1]=omega*u+ conjl(omega)*v-c2/3.;
            roots[2]= conjl(omega)*u+omega*v-c2/3.;
            return 0;
        }
    }
    LZ a[100]={0};
    for (int i=0;i<n;++i) {
        a[i*10+i]=alpha[i];
        if (i) a[(i-1)*10+i]=a[i*10+i-1]= csqrtl(beta[i]);
    }
    for (int size=n;size>1;--size) {
        int iteration;
        for (iteration=0;iteration<300;++iteration) {
            int last=size-1;
            if ( cabsl(a[last*10+last-1]) < (32*LDBL_EPSILON)*(1+ cabsl(a[last*10+last])+ cabsl(a[(last-1)*10+last-1])))
                break;
            LZ x=a[(last-1)*10+last-1], z=a[last*10+last];
            LZ disc= csqrtl((x-z)*(x-z)+4.*a[(last-1)*10+last]*a[last*10+last-1]);
            LZ shift1=(x+z+disc)*.5, shift2=(x+z-disc)*.5;
            LZ shift= cabsl(shift1-z)< cabsl(shift2-z)?shift1:shift2;
            LZ q[100]={0}, r[100];
            memcpy(r,a,sizeof(r));
            for (int i=0;i<size;++i) { r[i*10+i]-=shift; q[i*10+i]=1; }
            for (int k=0;k<size-1;++k) {
                LZ v[10]={0}; long double norm=0;
                for (int i=k;i<size;++i) { v[i]=r[i*10+k]; norm+= creall(v[i]* conjl(v[i])); }
                norm= sqrtl(norm);
                if (norm<DBL_MIN) continue;
                v[k]+= cabsl(v[k])>0?v[k]/ cabsl(v[k])*norm:norm;
                long double vnorm=0;
                for (int i=k;i<size;++i) vnorm+= creall(v[i]* conjl(v[i]));
                for (int j=k;j<size;++j) {
                    LZ dot=0;
                    for (int i=k;i<size;++i) dot+= conjl(v[i])*r[i*10+j];
                    dot*=2/vnorm;
                    for (int i=k;i<size;++i) r[i*10+j]-=v[i]*dot;
                }
                for (int i=0;i<size;++i) {
                    LZ dot=0;
                    for (int j=k;j<size;++j) dot+=q[i*10+j]*v[j];
                    dot*=2/vnorm;
                    for (int j=k;j<size;++j) q[i*10+j]-=dot* conjl(v[j]);
                }
            }
            for (int i=0;i<size;++i) for (int j=0;j<size;++j) {
                LZ value=0;
                for (int k=0;k<size;++k) value+=r[i*10+k]*q[k*10+j];
                a[i*10+j]=value+(i==j?shift:0);
            }
        }
        if (iteration==300) return GPW_NUMERIC;
        roots[size-1]=a[(size-1)*10+size-1];
    }
    roots[0]=a[0];
    return GPW_SUCCESS;
}

int gpw_rule(int n,Z t,Z *nodes,Z *weights,double *residual) {
    if (n<1||n>10) return GPW_RANGE;
    if (!isfinite(creal(t))||!isfinite(cimag(t))) return GPW_INPUT;
    /* Low-order analytic cases use a convergent small-|T| moment series.
     * This is a Rys rule, not an integral-backend fallback. Restrict the domain
     * to avoid cancellation of clustered high-|T| power moments. */
    if (n<=2 && cabs(t)<2.) {
        Z moments[4]={0},term=exp(fmin(creal(t),0.));
        for (int j=0;j<50;++j) {
            for (int m=0;m<2*n;++m) moments[m]+=term/(2*m+2.*j+1);
            term*=-t/(j+1.);
            if (cabs(term)<2e-17) break;
        }
        if (cabs(moments[0])<DBL_MIN) return GPW_NUMERIC;
        if (n==1) { nodes[0]=moments[1]/moments[0]; weights[0]=moments[0]; }
        else {
            Z a=moments[1]/moments[0];
            Z norm=moments[2]-2.*a*moments[1]+a*a*moments[0];
            if (cabs(norm)<1e-14*cabs(moments[0])) return GPW_NUMERIC;
            Z b=(moments[3]-2.*a*moments[2]+a*a*moments[1])/norm;
            Z delta=csqrt((a-b)*(a-b)+4.*norm/moments[0]);
            if (cabs(delta)<1e-14) return GPW_NUMERIC;
            nodes[0]=(a+b+delta)*.5; nodes[1]=(a+b-delta)*.5;
            weights[0]=(moments[1]-nodes[1]*moments[0])/delta;
            weights[1]=moments[0]-weights[0];
        }
        double worst=0;
        for (int m=0;m<2*n;++m) {
            Z v=0;
            for (int i=0;i<n;++i) v+=weights[i]*cpow(nodes[i],m);
            double err=cabs(v-moments[m])/(cabs(moments[m])+DBL_MIN);
            if (!isfinite(err)) return GPW_NUMERIC;
            worst=fmax(worst,err);
        }
        *residual=worst;
        return worst<2e-12?0:GPW_NUMERIC;
    }
    Measure m;
    int status=measure(t,&m);
    if (status) return status;
    if (n==2) {
        /* Near a zero of the complex zeroth moment, the two Stieltjes
         * diagonal entries become large and opposite. Form the monic
         * polynomial directly from the 2x2 moment system to avoid that
         * cancellation. This also avoids division by the zeroth moment. */
        LZ f[4]={0},g[4],roots[2]; long double bounds[4]={0},amplitude=0;
        for (int j=0;j<m.n;++j) {
            LZ v=m.w[j];
            for (int k=0;k<4;++k) { f[k]+=v; bounds[k]+=cabsl(v); v*=m.y[j]; }
        }
        for (int k=0;k<4;++k) amplitude=fmaxl(amplitude,cabsl(f[k]));
        if (!(amplitude>DBL_MIN)||!isfinite(amplitude)) return GPW_NUMERIC;
        for (int k=0;k<4;++k) g[k]=f[k]/amplitude;
        LZ det=g[0]*g[2]-g[1]*g[1];
        if (cabsl(det)<1e-13*(cabsl(g[0]*g[2])+cabsl(g[1]*g[1]))) return GPW_NUMERIC;
        LZ c1=(g[1]*g[2]-g[0]*g[3])/det,c0=(g[1]*g[3]-g[2]*g[2])/det;
        LZ disc=csqrtl(c1*c1-4*c0),plus=-c1+disc,minus=-c1-disc;
        roots[0]=.5*(cabsl(plus)>cabsl(minus)?plus:minus);
        if (cabsl(roots[0])<DBL_MIN) return GPW_NUMERIC;
        roots[1]=c0/roots[0];
        if (cabsl(roots[0]-roots[1])<1e-14*(1+cabsl(roots[0])+cabsl(roots[1]))) return GPW_NUMERIC;
        weights[0]=(f[1]-roots[1]*f[0])/(roots[0]-roots[1]);
        weights[1]=f[0]-weights[0];
        for (int i=0;i<2;++i) {
            nodes[i]=m.offset+m.factor*roots[i];
            if (!isfinite(creal(nodes[i]))||!isfinite(cimag(nodes[i]))||
                !isfinite(creal(weights[i]))||!isfinite(cimag(weights[i]))) return GPW_NUMERIC;
        }
        long double worst=0; LZ powers[2]={1,1};
        for (int k=0;k<4;++k) {
            long double err=cabsl(weights[0]*powers[0]+weights[1]*powers[1]-f[k])/fmaxl(bounds[k],DBL_MIN);
            if (!isfinite(err)) return GPW_NUMERIC;
            worst=fmaxl(worst,err);
            for (int i=0;i<2;++i) powers[i]*=roots[i];
        }
        *residual=worst;
        return worst<2e-11?GPW_SUCCESS:GPW_NUMERIC;
    }
    LZ p[GPW_MAXPOINTS],old[GPW_MAXPOINTS];
    LZ alpha[10], beta[10]={0}, norms[10];
    for (int j=0;j<m.n;++j) { p[j]=1; old[j]=0; }
    long double absnorm=0;
    for (int k=0;k<n;++k) {
        LZ norm=0,first=0; absnorm=0;
        for (int j=0;j<m.n;++j) {
            LZ v=m.w[j]*p[j]*p[j]; norm+=v; first+=m.y[j]*v; absnorm+=cabsl(v);
        }
        if (!isfinite(absnorm)||!isfinite(creall(norm))||!isfinite(cimagl(norm))||
            cabsl(norm)<DBL_MIN || cabsl(norm)<1e-13*absnorm) return GPW_NUMERIC;
        norms[k]=norm; alpha[k]=first/norm;
        if (k) beta[k]=norm/norms[k-1];
        if (k+1<n) for (int j=0;j<m.n;++j) {
            LZ next=(m.y[j]-alpha[k])*p[j]-beta[k]*old[j];
            old[j]=p[j]; p[j]=next;
        }
    }
    LZ roots[10];
    status=eigenvalues(n,alpha,beta,roots);
    if (status) return status;
    for (int i=0;i<n;++i) {
        LZ value=1,prev=0,denom=1;
        for (int k=1;k<n;++k) {
            LZ next=(roots[i]-alpha[k-1])*value-beta[k-1]*prev;
            prev=value; value=next;
            denom+=value*value*norms[0]/norms[k];
        }
        weights[i]=norms[0]/denom;
        nodes[i]=m.offset+m.factor*roots[i];
        if (!isfinite(creall(weights[i]))||!isfinite(cimagl(weights[i]))||
            !isfinite(creall(nodes[i]))||!isfinite(cimagl(nodes[i]))) return GPW_NUMERIC;
    }
    /* Check scaled y moments, not clustered s powers alone. */
    long double worst=0;
    LZ powers[10];
    for (int i=0;i<n;++i) powers[i]=1;
    for (int j=0;j<m.n;++j) p[j]=1;
    for (int k=0;k<2*n;++k) {
        LZ exact=0,rebuild=0; long double bound=0;
        for (int j=0;j<m.n;++j) { LZ v=m.w[j]*p[j]; exact+=v; bound+=cabsl(v); p[j]*=m.y[j]; }
        for (int i=0;i<n;++i) { rebuild+=weights[i]*powers[i]; powers[i]*=roots[i]; }
        long double err=cabsl(exact-rebuild)/fmaxl(bound,DBL_MIN);
        if (!isfinite(err)||!isfinite(bound)) return GPW_NUMERIC;
        worst=fmaxl(worst,err);
    }
    *residual=worst;
    return worst<2e-11?GPW_SUCCESS:GPW_NUMERIC;
}

int gpw_rys_rule(int32_t n,double tr,double ti,double *nodes,double *weights,
                 double *scale,double *residual) {
    if (!nodes||!weights||!scale||!residual) return GPW_INPUT;
    Z s[10],w[10];
    int status=gpw_rule(n,tr+I*ti,s,w,residual);
    if (status) return status;
    *scale=fmin(tr,0.);
    for (int i=0;i<n;++i) {
        nodes[2*i]=creal(s[i]); nodes[2*i+1]=cimag(s[i]);
        weights[2*i]=creal(w[i]); weights[2*i+1]=cimag(w[i]);
    }
    return 0;
}

int gpw_boys_scaled(int32_t order,double tr,double ti,double *out) {
    if (!out||order<0||order>19) return GPW_INPUT;
    Measure m; int status=measure(tr+I*ti,&m);
    if (status) return status;
    Z f[20]={0};
    for (int j=0;j<m.n;++j) {
        double s=m.offset+m.factor*m.y[j]; Z v=m.w[j];
        for (int i=0;i<=order;++i) { f[i]+=v; v*=s; }
    }
    for (int i=0;i<=order;++i) { out[2*i]=creal(f[i]); out[2*i+1]=cimag(f[i]); }
    return 0;
}
