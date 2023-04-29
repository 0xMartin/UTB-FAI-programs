/*
 * Ukazkovy program pro Programovani mikropocitacu
 * Ovladac pro SysTick timer.
 *
 * Detailni dokumentace je v .h souboru.
 *
 */

#include "MKL25Z4.h"
#include "drv_systick.h"



/* Example Internal functions */
/*static void _setbaudrate(UART0_baudrate baudrate);*/

/* Example of internal constants*/

/* The number in SysTick VAL register which equals to 1 us.
 * This is actually also the number of clock cycles which equals to 1 us.
 * We assume the F_CPU is always >= 1 MHz and so this value is always >= 1.
 * We assume SysTick interrupt every ms.
 * Example: F_CPU = 8 MHz, the counter counts from 8000 to 0 to generate interrupt
 * every ms. The MSF_SYSTICK_VALINUS = 8
 * Note1: that we do not need to use #if for F_CPU since CMSIS provides F_CPU in a variable.
 * Note2: this will not be exact for F_CPU which is not in whole MHz*/
#define		MSF_SYSTICK_VALINUS		(SystemCoreClock/1000000u)


#if (DEFAULT_SYSTEM_CLOCK == 20971520u)
	#undef MSF_SYSTICK_VALINUS
	#define MSF_SYSTICK_VALINUS		(21)	/* 21 is much better than 20 computed by F_CPU/x with integers */
#endif

/*
 Hold the number of ms elapsed from start of the program.
 It is incremented in SysTick interrupt every ms. The SysTick is
 initialized to interrupt every ms in systick_initialize()
*/
volatile uint32_t   gmsf_systime;
volatile uint32_t   gmsf_delaycnt;


/* Initialize the systick driver.*/
void SYSTICK_initialize(void)
{
	/* Use SysTick as reference for the delay loops.
	   Configure the SysTick interrupt to occur every ms */
	 if ( SysTick_Config (SystemCoreClock / 1000u ) != 0 )
	    	while(1)
	    		;	/* ERROR initializing SysTick timer */
}

/* Get the number of milliseconds that elapsed since the CPU started.  */
uint32_t SYSTICK_millis(void)
{
	return gmsf_systime;
}


/* Get the number of microseconds that elapsed since the CPU started. */
uint32_t SYSTICK_micros(void)
{
	uint32_t fraction = (SysTick->LOAD - SysTick->VAL) / MSF_SYSTICK_VALINUS;
	/* The value in the SysTick->LOAD register is the number of clock ticks
	 * between SysTick interrupts. The VAL register is loaded with this value
	 * and then decremented. When it reaches 0, SysTick interrupt occurs.
	 * initialize() sets the interrupt to occur every ms, so the value in
	 * LOAD is (SystemCoreClock / 1000) and this is how many "ticks" there are
	 * in 1 ms.
	 * msf_<device>.h defines constant we use to convert the count in LOAD to us  */

	    return (SYSTICK_millis() * 1000u + fraction);
}


/* Stop the execution for given number of milliseconds using busy-wait loop. */
void SYSTICK_delay_ms(uint32_t millis)
{
	 gmsf_delaycnt = millis;

	 /* Busy wait until the SysTick decrements the counter to zero */
	 while (gmsf_delaycnt != 0u)
		 ;
}

/* Handler for the SysTick interrupt.
The name of the function is pre-defined by CMSIS */
void SysTick_Handler (void)
{
    /* global system time */
    gmsf_systime++;

    /* Decrement to zero the counter used by the msf_delay_ms */
    if (gmsf_delaycnt != 0u)
    {
        gmsf_delaycnt--;
    }
}


 /*internal function */
/*
static void uart0_setbaudrate(UART0_baudrate baudrate)
{
}
*/
