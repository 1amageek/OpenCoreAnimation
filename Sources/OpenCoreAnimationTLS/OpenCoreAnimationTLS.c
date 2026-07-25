#include "OpenCoreAnimationTLS.h"

#include <errno.h>
#include <pthread.h>

static pthread_key_t transaction_stack_key;
static pthread_mutex_t initialization_mutex = PTHREAD_MUTEX_INITIALIZER;
static oca_tls_destructor_t registered_destructor = 0;
static int initialization_status = EAGAIN;
static int is_initialized = 0;

/*
 * Initialization is called through one synchronized Swift global before any
 * get/set operation can execute. After the mutex-protected initialization
 * completes, the status fields are immutable for the process lifetime.
 */
int oca_tls_initialize(oca_tls_destructor_t destructor) {
    int lock_status = pthread_mutex_lock(&initialization_mutex);
    if (lock_status != 0) {
        return lock_status;
    }

    if (is_initialized) {
        int result = registered_destructor == destructor
            ? initialization_status
            : EINVAL;
        pthread_mutex_unlock(&initialization_mutex);
        return result;
    }

    initialization_status = pthread_key_create(
        &transaction_stack_key,
        destructor
    );
    if (initialization_status == 0) {
        registered_destructor = destructor;
    }
    is_initialized = 1;

    int unlock_status = pthread_mutex_unlock(&initialization_mutex);
    if (initialization_status != 0) {
        return initialization_status;
    }
    return unlock_status;
}

void *oca_tls_get(void) {
    if (!is_initialized || initialization_status != 0) {
        return 0;
    }
    return pthread_getspecific(transaction_stack_key);
}

int oca_tls_set(void *value) {
    if (!is_initialized) {
        return EINVAL;
    }
    if (initialization_status != 0) {
        return initialization_status;
    }
    return pthread_setspecific(transaction_stack_key, value);
}
