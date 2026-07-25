#ifndef OPEN_CORE_ANIMATION_TLS_H
#define OPEN_CORE_ANIMATION_TLS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*oca_tls_destructor_t)(void *);

/*
 * The slot owns no allocation and does not inspect the value. The caller must
 * publish one retained owner and either clear and release it or allow the
 * registered destructor to release it at thread exit.
 */
int oca_tls_initialize(oca_tls_destructor_t destructor);
void *oca_tls_get(void);
int oca_tls_set(void *value);

#ifdef __cplusplus
}
#endif

#endif
