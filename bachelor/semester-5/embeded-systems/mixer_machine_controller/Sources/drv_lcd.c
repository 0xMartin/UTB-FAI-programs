/*
 * Ukazkovy program pro Programovani mikropocitacu
 * Ovladac pro LCD display.
 *
 * Detailni dokumentace je v .h souboru.
 *
 */

#include "MKL25Z4.h"
#include "drv_lcd.h"


/* Pin abstraction */
#define	LCD_PINS_DB4_PIN		(8)		/* number of the pin on MCU port for DB4 display pin  */
#define	LCD_PINS_DB5_PIN		(9)
#define	LCD_PINS_DB6_PIN		(10)
#define	LCD_PINS_DB7_PIN		(11)
#define	LCD_PINS_DATA_PORT		(PORTC)	/* port for the data pins of the lcd */
#define LCD_PINS_DATA_GPIO		(PTC)	/* gpio struct for data pins */

#define	LCD_PINS_E_PIN			(12)
#define	LCD_PINS_E_PORT			(PORTC)
#define	LCD_PINS_E_GPIO			(PTC)

#define	LCD_PINS_RS_PIN			(13)
#define	LCD_PINS_RS_PORT		(PORTC)
#define	LCD_PINS_RS_GPIO		(PTC)

#define	LCD_PINS_RW_PIN			(12)
#define	LCD_PINS_RW_PORT		(PORTA)
#define	LCD_PINS_RW_GPIO		(PTA)

#define	LCD_PINS_LIGHT_PIN		(13)
#define	LCD_PINS_LIGHT_PORT		(PORTA)
#define	LCD_PINS_LIGHT_GPIO		(PTA)

/* Mask used for enabling clock to ports.
 * Must include all the ports used by the pins defined above!*/
#define	LCD_PORT_CLOCK_MASK		(SIM_SCGC5_PORTA_MASK | SIM_SCGC5_PORTC_MASK)


/* Macros for pin operations */
#define	LCD_PinSet(gpio_struct, pin_number)			gpio_struct->PSOR |= (1 << pin_number)
#define	LCD_PinClear(gpio_struct, pin_number)		gpio_struct->PCOR |= (1 << pin_number)
#define	LCD_PinDirOutput(gpio_struct, pin_number)	gpio_struct->PDDR |= (1 << pin_number)
#define	LCD_PinDirInput(gpio_struct, pin_number)	gpio_struct->PDDR &= ~(1 << pin_number)
#define	LCD_PinRead(gpio_struct, pin_number)		((gpio_struct->PDIR & (1 << pin_number)) ? 1 : 0)

/* Convenience macros for data pins */
#define	LCD_DB4_Set()	LCD_PinSet(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN)
#define	LCD_DB5_Set()	LCD_PinSet(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN)
#define	LCD_DB6_Set()	LCD_PinSet(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN)
#define	LCD_DB7_Set()	LCD_PinSet(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN)
#define	LCD_DB4_Clear()	LCD_PinClear(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN)
#define	LCD_DB5_Clear()	LCD_PinClear(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN)
#define	LCD_DB6_Clear()	LCD_PinClear(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN)
#define	LCD_DB7_Clear()	LCD_PinClear(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN)
#define	LCD_DB4_Read()	LCD_PinRead(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN)
#define	LCD_DB5_Read()	LCD_PinRead(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN)
#define	LCD_DB6_Read()	LCD_PinRead(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN)
#define	LCD_DB7_Read()	LCD_PinRead(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN)

#define lcd_short_delay() __asm__ __volatile__ ("NOP" "\n\t")
static const int F_LCD_CHARS_MAX = 20;

/* Internal functions */
static inline void lcd_delay_1us(void) __attribute__((always_inline));
/* Inline functions for manipulating control pins */
static inline void lcd_rs_put_val(uint8_t value);
static inline void lcd_rw_put_val(uint8_t value);
static inline void lcd_e_put_val(uint8_t value);
/* manipulate data pins */
static inline void lcd_db4_put_val(uint8_t value);
static inline void lcd_db5_put_val(uint8_t value);
static inline void lcd_db6_put_val(uint8_t value);
static inline void lcd_db7_put_val(uint8_t value);
static inline uint8_t lcd_db4_get_val(void);
static inline uint8_t lcd_db5_get_val(void);
static inline uint8_t lcd_db6_get_val(void);
static inline uint8_t lcd_db7_get_val(void);

static void lcd_wait_bf(void);
static void lcd_wr_data(uint8_t value);
static void lcd_wr_register(uint8_t value);
static void lcd_delay_us(uint32_t microseconds);
static void lcd_pulse_e_pin(void);

/* Example of internal constants*/
/*const char UTB_UART_CR = (const char)0x0D;*/


/* initialize display */
void LCD_initialize(void)
{
	/* enable clock for ports used by this driver */
	SIM->SCGC5 |= LCD_PORT_CLOCK_MASK;

	/* Wait for power up of the display - 20 ms */
	lcd_delay_us(21000);

	/* Set pin function to GPIO for all used pins*/
	LCD_PINS_DATA_PORT->PCR[LCD_PINS_DB4_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_DATA_PORT->PCR[LCD_PINS_DB5_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_DATA_PORT->PCR[LCD_PINS_DB6_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_DATA_PORT->PCR[LCD_PINS_DB7_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_E_PORT->PCR[LCD_PINS_E_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_RS_PORT->PCR[LCD_PINS_RS_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_RW_PORT->PCR[LCD_PINS_RW_PIN] = PORT_PCR_MUX(1);
	LCD_PINS_LIGHT_PORT->PCR[LCD_PINS_LIGHT_PIN] = PORT_PCR_MUX(1);


	/* Set direction for output pins */
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN);
	LCD_PinDirOutput(LCD_PINS_E_GPIO, LCD_PINS_E_PIN);
	LCD_PinDirOutput(LCD_PINS_RS_GPIO, LCD_PINS_RS_PIN);
	LCD_PinDirOutput(LCD_PINS_RW_GPIO, LCD_PINS_RW_PIN);
	LCD_PinDirOutput(LCD_PINS_LIGHT_GPIO, LCD_PINS_LIGHT_PIN);

	// back light off
	LCD_PinClear(LCD_PINS_LIGHT_GPIO, LCD_PINS_LIGHT_PIN);

	lcd_rs_put_val(0);
	lcd_rw_put_val(0);
	lcd_db7_put_val(0);
	lcd_db6_put_val(0);
	lcd_db5_put_val(1);
	lcd_db4_put_val(1);

	lcd_pulse_e_pin();
	lcd_delay_us(4500);      // cekej > 4.1 ms

	lcd_pulse_e_pin();
	lcd_delay_us(150);      // cekej > 100 us

	lcd_pulse_e_pin();
	lcd_delay_us(150);      // cekej > 100 us

	lcd_db7_put_val(0);
	lcd_db6_put_val(0);
	lcd_db5_put_val(1);
	lcd_db4_put_val(0);

	lcd_pulse_e_pin();
	lcd_delay_us(150);      // cekej > 100 us

	// ------------------------------------------------
	// Nastaveni 4 bit rozhrani, 2 radky, 5x8 font
	// N = 1
	// F = 0
	lcd_wr_register(0b00101000);

	// ------------------------------------------------
	// 2) Display ON/OFF control
	// ------------------------------------------------
	// D = 1 - zapnuti displeje
	// C = 0 - kurzor vypnut
	// B = 0 - blikani znaku vypnuto
	lcd_wr_register(0b00001100);

	// ------------------------------------------------
	// 3) Display clear
	// ------------------------------------------------
	lcd_wr_register(0b00000001);

	// ------------------------------------------------
	// 4) Entry mode set
	// ------------------------------------------------
	// I/D = 1, S/H = 0
	lcd_wr_register(0b00000110);


}

/* Set cursor to given line and column within the line */
void LCD_set_cursor(uint8_t line, uint8_t column)
{
	uint8_t address;

	if (column < 1) {
		column = 1;
	}

	if (column > F_LCD_CHARS_MAX) {
		column = F_LCD_CHARS_MAX;
	}

	if (line < 1) {
		line = 1;
	}

	if (line > 4) {
		line = 4;
	}

	switch (line) {
	case 1:
		address = column - 1;
		break;
	case 2:
		address = 63 + column;
		break;
	case 3:
		address = 19 + column;
		break;
	case 4:
		address = 83 + column;
		break;
	}

	lcd_wr_register(0b10000000 | address);
}

/* Display one character on the display; at current cursor position. */
void LCD_putch(char c)
{
	lcd_wr_data(c);
}

/* Display null-terminated string on the display; at current cursor position. */
void LCD_puts(const char* str)
{
	uint8_t n = 0;
	while(*str) {
		LCD_putch(*str);
		str++;
		n++;
		if (n > F_LCD_CHARS_MAX) {
			// chyba: textovy retezec je prilis dlouhy nebo neni zakoncen nulou
			while(1)
				;
		}
	}
}

/* Clear the display. */
void LCD_clear(void)
{
	// smazani displeje
	lcd_wr_register(0b00000001);
	// kurzor home
	lcd_wr_register(0b00000010);
}

void LCD_backlight_on(void)
{
	// back light off
	LCD_PinSet(LCD_PINS_LIGHT_GPIO, LCD_PINS_LIGHT_PIN);
}

void LCD_backlight_off(void)
{
	// back light off
	LCD_PinClear(LCD_PINS_LIGHT_GPIO, LCD_PINS_LIGHT_PIN);
}





 /*internal functions */

/* Fixed delay of 1 microsecond at 48 MHz clock */
static inline void lcd_delay_1us(void)
{
    // Need 48 clocks
    // For N repetitions the loop takes 3*N-1 clocks
    // plus 5 clocks the other code;
    // 3 * 14 -1 = 41 + 5 = 46 -> add 2 nops
    __asm__ __volatile__ (
    	"NOP" "\n\t"     // 1 clock
        "NOP" "\n\t"
        "PUSH {r0}" "\n\t"			// 2 clocks
        "MOV r0,#14" "\n\t"         // 1 clock
        "1: SUB r0, #1" "\n\t"	// 1 clock
		"BNE 1b"  "\n\t"		// 2 clocks if branches, 1 if not
		"POP {r0}"              // 2 clocks
	);
}

static inline void lcd_rs_put_val(uint8_t value)
{
	if (value)
		LCD_PinSet(LCD_PINS_RS_GPIO,LCD_PINS_RS_PIN);
	else
		LCD_PinClear(LCD_PINS_RS_GPIO, LCD_PINS_RS_PIN);
}

static inline void lcd_rw_put_val(uint8_t value)
{
	if (value)
		LCD_PinSet(LCD_PINS_RW_GPIO, LCD_PINS_RW_PIN);
	else
		LCD_PinClear(LCD_PINS_RW_GPIO, LCD_PINS_RW_PIN);
}

static inline void lcd_e_put_val(uint8_t value)
{
	if (value)
		LCD_PinSet(LCD_PINS_E_GPIO, LCD_PINS_E_PIN);
	else
		LCD_PinClear(LCD_PINS_E_GPIO, LCD_PINS_E_PIN);
}

static inline void lcd_db4_put_val(uint8_t value)
{
	if (value)
		LCD_DB4_Set();
	else
		LCD_DB4_Clear();
}

static inline void lcd_db5_put_val(uint8_t value)
{
	if (value)
		LCD_DB5_Set();
	else
		LCD_DB5_Clear();
}

static inline void lcd_db6_put_val(uint8_t value)
{
	if (value)
		LCD_DB6_Set();
	else
		LCD_DB6_Clear();
}

static inline void lcd_db7_put_val(uint8_t value)
{
	if (value)
		LCD_DB7_Set();
	else
		LCD_DB7_Clear();
}

static inline uint8_t lcd_db4_get_val(void)
{
	return LCD_DB4_Read();
}

static inline uint8_t lcd_db5_get_val(void)
{
	return LCD_DB5_Read();
}
static inline uint8_t lcd_db6_get_val(void)
{
	return LCD_DB6_Read();
}
static inline uint8_t lcd_db7_get_val(void)
{
	return LCD_DB7_Read();
}



static void lcd_delay_us(uint32_t microseconds)
{
	while (microseconds > 0 )
	{
		lcd_delay_1us();
		microseconds--;
	}
}

/* Generate short pulse on pin E */
static void lcd_pulse_e_pin(void)
{
	lcd_e_put_val(1);
	lcd_short_delay();
	lcd_e_put_val(0);
	lcd_short_delay();

}


static void lcd_wr_register(uint8_t value)
{
    // DB4-DB7 vystupni rezim
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN);

	lcd_rs_put_val(0);
	lcd_rw_put_val(0);

	lcd_db7_put_val((value >> 7) & 1);
	lcd_db6_put_val((value >> 6) & 1);
	lcd_db5_put_val((value >> 5) & 1);
	lcd_db4_put_val((value >> 4) & 1);
	lcd_pulse_e_pin();

    // ------------------------------
	lcd_db7_put_val((value >> 3) & 1);
	lcd_db6_put_val((value >> 2) & 1);
	lcd_db5_put_val((value >> 1) & 1);
	lcd_db4_put_val((value >> 0) & 1);
	lcd_pulse_e_pin();

    lcd_wait_bf();
}



static void lcd_wr_data(uint8_t value)
{
	// DB4-DB7 vystupni rezim
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN);
	LCD_PinDirOutput(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN);

	// ------------------------------
	lcd_rs_put_val(1);
	lcd_rw_put_val(0);

	lcd_db7_put_val((value >> 7) & 1);
	lcd_db6_put_val((value >> 6) & 1);
	lcd_db5_put_val((value >> 5) & 1);
	lcd_db4_put_val((value >> 4) & 1);
	lcd_pulse_e_pin();

	// ------------------------------
	lcd_rs_put_val(1);
	lcd_rw_put_val(0);
	lcd_db7_put_val((value >> 3) & 1);
	lcd_db6_put_val((value >> 2) & 1);
	lcd_db5_put_val((value >> 1) & 1);
	lcd_db4_put_val((value >> 0) & 1);
	lcd_pulse_e_pin();

	lcd_wait_bf();

}




static void lcd_wait_bf(void)
{
    uint8_t value;

    LCD_PinDirInput(LCD_PINS_DATA_GPIO, LCD_PINS_DB4_PIN);
    LCD_PinDirInput(LCD_PINS_DATA_GPIO, LCD_PINS_DB5_PIN);
    LCD_PinDirInput(LCD_PINS_DATA_GPIO, LCD_PINS_DB6_PIN);
    LCD_PinDirInput(LCD_PINS_DATA_GPIO, LCD_PINS_DB7_PIN);

    while(1)
    {
        // cteni registru s BF priznakem
        // DB4-DB7 vstupni rezim
    	lcd_rs_put_val(0);
    	lcd_rw_put_val(1);
    	lcd_e_put_val(1);
    	lcd_short_delay();

    	value = 0;
        // precteme horni 4 bity
    	if ( lcd_db7_get_val())
    		value |= (1 << 7);
    	if ( lcd_db6_get_val())
    	    value |= (1 << 6);
    	if ( lcd_db5_get_val())
    	    value |= (1 << 5);
    	if ( lcd_db4_get_val())
    	    value |= (1 << 4);

    	lcd_e_put_val(0);
    	lcd_short_delay();
    	lcd_e_put_val(1);
		lcd_short_delay();

		// precteme dolni 4 bity
		if (lcd_db7_get_val())
			value |= (1 << 3);
		if (lcd_db6_get_val())
			value |= (1 << 2);
		if (lcd_db5_get_val())
			value |= (1 << 1);
		if (lcd_db4_get_val())
			value |= (1 << 0);

		lcd_e_put_val(0);
		lcd_short_delay();

        // Test BF priznaku
        if ((value & 0b10000000) == 0)
        	break;
    }

}


