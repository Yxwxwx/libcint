#include "gpw.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int32_t atm[6]={2,20,1,0,0,0}, bas[8]={0,0,1,1,0,23,24,0};
    int32_t shls[3]={0,0,0};
    double env[26]={0},k[3]={0},out[2];
    env[23]=env[24]=1;
    const double pi=acos(-1.);
    GPWOpt *opt=NULL; GPWWorkspace *work=NULL;
    assert(gpw_abi_version()==1 && gpw_integer_bytes()==4);
    assert(gpw_opt_create(&opt,atm,1,bas,1,env,25)==0);
    assert(gpw_workspace_create(&work,1)==0);
    assert(gpw_int2e_gppg_cart(out,1,shls,k,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]-pow(pi,1.5)/(2*sqrt(2.)))<1e-13 && fabs(out[1])<1e-14);
    assert(gpw_int2e_gppg_sph(out,1,shls,k,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]-pow(pi,1.5)/(2*sqrt(2.)))<1e-13);
    assert(gpw_int2e_gpgg_cart(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]-pi/(8*sqrt(3.)))<1e-13 && fabs(out[1])<1e-14);
    assert(gpw_int2e_gpgg_sph(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]-pi/(8*sqrt(3.)))<1e-13);
    assert(gpw_int1e_ovlp_gp_cart(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]-pi/2)<1e-13 && out[1]==0);
    assert(gpw_int1e_ovlp_gp_sph(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]-pi/2)<1e-13);
    assert(gpw_int1e_kin_gp_cart(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(out[0]==0 && out[1]==0);
    assert(gpw_int1e_kin_gp_sph(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(out[0]==0 && out[1]==0);
    assert(gpw_int1e_nuc_gp_cart(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]+2*sqrt(pi))<1e-13 && fabs(out[1])<1e-14);
    assert(gpw_int1e_nuc_gp_sph(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==0);
    assert(fabs(out[0]+2*sqrt(pi))<1e-13);
    assert(gpw_int1e_batch(NULL,0,GPW_NUC,0,shls,NULL,0,atm,1,bas,1,env,25,opt,work)==0);
    out[0]=123;
    assert(gpw_int1e_nuc_gp_cart(out,0,shls,k,atm,1,bas,1,env,25,opt,work)==GPW_INPUT);
    assert(out[0]==123);
    assert(gpw_int2e_gpgg_sph(out,0,shls,k,atm,1,bas,1,env,25,opt,work)==GPW_INPUT);
    env[23]=2;
    assert(gpw_int2e_gpgg_sph(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==GPW_STALE_OPT);
    assert(gpw_int1e_ovlp_gp_cart(out,1,shls,k,atm,1,bas,1,env,25,opt,work)==GPW_STALE_OPT);
    env[23]=-1;
    assert(gpw_int2e_gpgg_sph(out,1,shls,k,atm,1,bas,1,env,25,NULL,work)==GPW_INPUT);
    gpw_workspace_destroy(work); gpw_opt_destroy(opt);
    /* A later numerical failure must not expose a successful partial batch. */
    double waves[6]={0,0,0,1e308,0,0}, failed[6]={123,7,7,7,7,456};
    env[23]=1.;
    assert(gpw_int2e_batch(failed+1,2,GPW_GPPG,1,shls,waves,waves,2,
                         atm,1,bas,1,env,25,NULL,NULL)<0);
    for (int i=1;i<5;++i) assert(failed[i]==0);
    assert(failed[0]==123 && failed[5]==456);
    for (int op=GPW_OVLP;op<=GPW_NUC;++op) {
        for (int i=1;i<5;++i) failed[i]=7;
        assert(gpw_int1e_batch(failed+1,2,op,1,shls,waves,2,atm,1,bas,1,env,25,NULL,NULL)<0);
        for (int i=1;i<5;++i) assert(failed[i]==0);
        assert(failed[0]==123 && failed[5]==456);
    }
    atm[2]=99;
    assert(gpw_int1e_nuc_gp_cart(out,1,shls,k,atm,1,bas,1,env,25,NULL,NULL)==GPW_RANGE);
    atm[2]=2; atm[3]=26;
    assert(gpw_int1e_nuc_gp_cart(out,1,shls,k,atm,1,bas,1,env,25,NULL,NULL)==GPW_INPUT);
    atm[2]=1; atm[3]=0;
    env[23]=1.; env[25]=.3;
    bas[1]=6; bas[3]=2;
    size_t dim=56,n=dim*dim*dim;
    double *tensor=malloc((2*n+2)*sizeof(double));
    assert(tensor);
    tensor[0]=123.; tensor[2*n+1]=456.;
    k[0]=.3; k[1]=-.2; k[2]=.1;
    for (int repeat=0;repeat<8;++repeat) {
        assert(gpw_opt_create(&opt,atm,1,bas,1,env,26)==0);
        assert(gpw_workspace_create(&work,n)==0);
        assert(gpw_int2e_gpgg_cart(tensor+1,n,shls,k,atm,1,bas,1,env,26,opt,work)==0);
        for (size_t i=0;i<2*n;++i) assert(isfinite(tensor[i+1]));
        assert(fabs(tensor[1+2*(28*dim*dim)]-.3*tensor[1])<1e-11);
        assert(gpw_int2e_gpgg_sph(tensor+1,n,shls,k,atm,1,bas,1,env,26,opt,work)==0);
        assert(gpw_int2e_gppg_cart(tensor+1,n,shls,k,k,atm,1,bas,1,env,26,opt,work)==0);
        assert(gpw_int2e_gppg_sph(tensor+1,n,shls,k,k,atm,1,bas,1,env,26,opt,work)==0);
        for (int op=GPW_OVLP;op<=GPW_NUC;++op) for (int cart=0;cart<=1;++cart) {
            size_t size=cart?56:26;
            assert(gpw_int1e_batch(tensor+1,n,op,cart,shls,k,1,atm,1,bas,1,env,26,opt,work)==0);
            for (size_t i=0;i<2*size;++i) assert(isfinite(tensor[i+1]));
            assert(fabs(tensor[1+size]-.3*tensor[1])<1e-11);
        }
        assert(tensor[0]==123. && tensor[2*n+1]==456.);
        gpw_workspace_destroy(work); gpw_opt_destroy(opt);
    }
    assert(gpw_workspace_create(&work,1)==0);
    assert(gpw_int2e_gpgg_cart(tensor+1,n,shls,k,atm,1,bas,1,env,26,NULL,work)==GPW_WORKSPACE);
    gpw_workspace_destroy(work);
    assert(gpw_workspace_create(&work,1)==0);
    assert(gpw_int1e_nuc_gp_cart(tensor+1,n,shls,k,atm,1,bas,1,env,26,NULL,work)==GPW_WORKSPACE);
    gpw_workspace_destroy(work);
    assert(gpw_int1e_nuc_gp_cart(env,n,shls,k,atm,1,bas,1,env,26,NULL,NULL)==GPW_INPUT);
    assert(gpw_int2e_gpgg_cart(env,n,shls,k,atm,1,bas,1,env,26,NULL,NULL)==GPW_INPUT);
    free(tensor);
    puts("GPW pure C smoke passed (ten entries, l=6, general contraction, capacity, opt and repeated workspaces)");
    return 0;
}
