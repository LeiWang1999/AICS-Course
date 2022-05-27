#if BANG_LOG == 1
#define __DEBUG_SCALAR
#endif

#define __RECORD_TIME__ 0
#ifdef __DEBUG_VECTOR
  #define PRINTF_VECTOR(statement, format, arr, len) \
    do {                                             \
      __bang_printf(statement);                      \
      __bang_printf("\n");                           \
      for (int idx = 0; idx < len; ++idx) {          \
        __bang_printf(format, *(arr + idx));         \
      }                                              \
      __bang_printf("\n\n");                         \
    } while (0)

#else
  #define PRINTF_VECTOR(format, ...)
#endif

#ifdef __DEBUG_SCALAR
  #define PRINTF_SCALAR(format, ...)                 \
    __bang_printf(format, ##__VA_ARGS__)
#else
  #define PRINTF_SCALAR(format, ...)
#endif
