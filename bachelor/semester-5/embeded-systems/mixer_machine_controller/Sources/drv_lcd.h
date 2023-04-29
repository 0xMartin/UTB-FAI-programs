/*
 * @file     drv_lcd.h
 * @brief    LCD display driver for Freescale KL25Z
 * @version  V1.00
 * @date     22. July 2015
 *
 * @note
 *
 * Ukazkovy program pro Programovani mikropocitacu
 * Ovladac pro LCD displej na kitu.
 *
 * Pouzite piny:
 * C8	- LCD DB4
 * C9	- LCD DB5
 * C10	- LCD DB6
 * C11	- LCD DB7
 * C12	- LCD E
 * C13	- LCD RS
 * A12	- LCD RW
 * A13	- LCD backlight control
 *
 */
#ifndef	UTBFRDM_DRV_LCD_H
#define UTBFRDM_DRV_LCD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


/** \ingroup  UTB_FRDM_Drivers
    \defgroup UTB_FRDM_LCDDriver LCD display driver for FRDM-KL25Z
  @{
 */


 /**
  * @brief Initialize the display
  * @return none
  * @note
  *
  */
void LCD_initialize(void);

/**
  * @brief Set cursor to given line and column within the line
  * @param line The line to set cursor to: 1 - 4
  * @param column The column within the line to set cursor to: 1 - 20
  * @return none
  * @note
  *
  */
void LCD_set_cursor(uint8_t line, uint8_t column);

/**
  * @brief Display one character on the display; at current cursor position.
  * @param c Character to display
  * @return none
  * @note
  */
void LCD_putch(char c);

/**
  * @brief Display null-terminated string on the display; at current cursor position.
  * @param str String to display
  * @return none
  * @note
  */
void LCD_puts(const char* str);

/**
  * @brief Clear the display.
  * @return none
  * @note
  */
void LCD_clear(void);

/**
  * @brief Turn on the display back light
  * @return none
  * @note
  */
void LCD_backlight_on(void);

/**
  * @brief Turn off the display back light
  * @return none
  * @note
  */
void LCD_backlight_off(void);


/*@} end of UTB_FRDM_LCDDriver */


#ifdef __cplusplus
}
#endif

#endif	/* UTBFRDM_DRV_LCD_H */

