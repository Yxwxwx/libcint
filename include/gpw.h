#ifndef GPW_H
#define GPW_H
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
# define GPW_API __declspec(dllexport)
#else
# define GPW_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPWOpt GPWOpt;
typedef struct GPWWorkspace GPWWorkspace;
enum { GPW_SUCCESS=0, GPW_SCREENED=1, GPW_INPUT=-1, GPW_RANGE=-2,
       GPW_NUMERIC=-3, GPW_MEMORY=-4, GPW_WORKSPACE=-5, GPW_ABI=-6,
       GPW_STALE_OPT=-7 };
enum { GPW_GPPG=2, GPW_GPGG=3 };
enum { GPW_OVLP=1, GPW_KIN=2, GPW_NUC=3 };

GPW_API int gpw_abi_version(void);
GPW_API int gpw_integer_bytes(void);
GPW_API const char *gpw_build_version(void);
GPW_API const char *gpw_error_string(int status);
GPW_API int gpw_opt_create(GPWOpt **opt, const int32_t *atm, int32_t natm,
    const int32_t *bas, int32_t nbas, const double *env, size_t nenv);
GPW_API void gpw_opt_destroy(GPWOpt *opt);
/* Capacity is in complex Cartesian shell elements; each call needs two buffers.
 * A workspace belongs to one concurrent caller; an opt is immutable/shareable. */
GPW_API int gpw_workspace_create(GPWWorkspace **work, size_t capacity);
GPW_API void gpw_workspace_destroy(GPWWorkspace *work);
GPW_API size_t gpw_workspace_bytes(const GPWWorkspace *work);
GPW_API uint64_t gpw_workspace_rules(const GPWWorkspace *work);
GPW_API uint64_t gpw_workspace_special(const GPWWorkspace *work);

/* <G_a|O|exp(+i k.r)> in atomic units. The plane wave is unnormalized.
 * O = 1, -laplacian/2, or the sum of nuclear attraction potentials.
 * Uses libcint's point, Gaussian and fractional-charge nuclear models.
 * Output order: contraction, Cartesian/spherical component (C order).
 * shls contains one real GTO shell; no dummy PW shell is needed. */
GPW_API int gpw_int1e_ovlp_gp_cart(double *out_ri, size_t out_ncomplex,
    const int32_t shls[1], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int1e_ovlp_gp_sph(double *out_ri, size_t out_ncomplex,
    const int32_t shls[1], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int1e_kin_gp_cart(double *out_ri, size_t out_ncomplex,
    const int32_t shls[1], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int1e_kin_gp_sph(double *out_ri, size_t out_ncomplex,
    const int32_t shls[1], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int1e_nuc_gp_cart(double *out_ri, size_t out_ncomplex,
    const int32_t shls[1], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int1e_nuc_gp_sph(double *out_ri, size_t out_ncomplex,
    const int32_t shls[1], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);

/* k is flat (count,3); output is (count,nAO_shell). op is GPW_OVLP/KIN/NUC.
 * The same capacity, opt, workspace and failure rules as int2e_batch apply. */
GPW_API int gpw_int1e_batch(double *out_ri, size_t out_ncomplex,
    int32_t op, int32_t cart, const int32_t shls[1], const double *k, size_t count,
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);

GPW_API int gpw_int2e_gpgg_cart(double *out_ri, size_t out_ncomplex,
    const int32_t shls[3], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int2e_gppg_cart(double *out_ri, size_t out_ncomplex,
    const int32_t shls[2], const double k[3], const double kp[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int2e_gpgg_sph(double *out_ri, size_t out_ncomplex,
    const int32_t shls[3], const double k[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);
GPW_API int gpw_int2e_gppg_sph(double *out_ri, size_t out_ncomplex,
    const int32_t shls[2], const double k[3], const double kp[3],
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);

/* shls has `kind` real GTO indices. k/kp are flat (count,3), zip paired.
 * cart is 1 or 0. count=0 is valid. No MD fallback exists in this library.
 * Validation failures leave output untouched. A failure after calculation
 * starts clears the entire requested output; the negative status is mandatory
 * to check. A failed batch never exposes partially computed integrals. */
GPW_API int gpw_int2e_batch(double *out_ri, size_t out_ncomplex,
    int32_t kind, int32_t cart, const int32_t *shls,
    const double *k, const double *kp, size_t count,
    const int32_t *atm, int32_t natm, const int32_t *bas, int32_t nbas,
    const double *env, size_t nenv, const GPWOpt *opt, GPWWorkspace *work);

/* Diagnostics: weights reconstruct exp(scale)*F_m(T), nodes are s=t^2.
 * nodes_ri and weights_ri each have 2*n double elements. */
GPW_API int gpw_rys_rule(int32_t n, double tr, double ti, double *nodes_ri,
    double *weights_ri, double *scale, double *residual);
GPW_API int gpw_boys_scaled(int32_t order, double tr, double ti, double *out_ri);

#ifdef __cplusplus
}
#endif
#endif
