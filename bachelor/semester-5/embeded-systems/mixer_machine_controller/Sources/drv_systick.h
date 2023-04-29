/*
 * @file     drv_systick.h
 * @brief    System timer SysTick driver for Freescale KL25Z
 * @version  V1.00
 * @date     17. July 2015
 *
 * @note	Provides functions for measuring time and precisely timed delay.
 *
 * Ukazkovy program pro Programovani mikropocitacu
 * Ovladac pro mereni casu a zpozdeni s nastavitelnou delkou generovate casovacem SysTick.
 *

 *
 */
#ifndef	UTBFRDM_DRV_SYSTICK_H
#define UTBFRDM_DRV_SYSTICK_H

#ifdef __cplusplus
extern "C" {
#endif


/** \ingroup  UTB_FRDM_Drivers
    \defgroup UTB_FRDM_SysTickDriver SysTick Driver for FRDM-KL25Z
  @{
 */


 /**
  * @brief Initialize the systick driver.
  * @return none
  * @note Sets up the SysTick timer to tick every millisecond.
  *
  */
void SYSTICK_initialize(void);

/**
  * @brief Get the number of milliseconds that elapsed since the CPU started.
  * @return number of milliseconds as 32-bit unsigned integer.
  * @note The value overflows in about 49.7 days. (2^32 milliseconds)
  */
uint32_t SYSTICK_millis(void);


/**
 * @brief Get the number of microseconds that elapsed since the CPU started.
 * @return number of microseconds as 32-bit unsigned integer.
 * @note  The value overflows in about 71.5 minutes.
 */
uint32_t SYSTICK_micros(void);


/**
 * @brief Stop the execution for given number of milliseconds using busy-wait loop.
 * @param  millis number of milliseconds for which the program should wait
 * @return none
 * @note Uses CMSIS SysTick timer interrupt.
 */
void SYSTICK_delay_ms(uint32_t millis);


/*@} end of UTB_FRDM_SysTickDriver */


#ifdef __cplusplus
}
#endif

#endif	/* UTBFRDM_DRV_SYSTICK_H */

