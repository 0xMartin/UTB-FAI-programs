#include "MKL25Z4.h"
#include "mixer.h"
#include "drv_lcd.h"
#include <stdlib.h>

#define ASYNCHRONOUS
#define SWITCH_PIN (4) // A4
#define RESET_PIN (5) // A5

static int i = 0;
static int state = 0;

// Prototype
static void init(void);
static inline int IsKeyPressed(int pin);
static void ADCInit(void);
static uint32_t ADCCalibrate(void);
static uint16_t read_analog_value(int min, int max);

static void display_main(uint16_t mixing_time);
static void display_run();
static void display_done();
static void display_reset();

int main(void) {
	MIXER_init();
	init();

	uint32_t wait;
	uint16_t val;
	uint16_t nval;

	for (;;) {
#ifdef ASYNCHRONOUS

		//reset
		if (IsKeyPressed(RESET_PIN) && state < 12) {
			state = 12;
		}

		switch (state) {
		//wait for run
		case 0:
			//dipslay main screen
			nval = read_analog_value(1, 10);
			if (nval != val) {
				display_main(nval);
				i = 0;
			}
			val = nval;
			//run
			if (IsKeyPressed(SWITCH_PIN)) {
				display_run();
				state = 1;
			}
			break;

			//process
		case 1:
			if (MIXER_fillTank1(H2_Level)) {
				state = 2;
			}
			break;
		case 2:
			if (MIXER_fillTank2(H4_Level)) {
				state = 3;
			}
			break;
		case 3:
			if (MIXER_fillTank3(H8_Level)) {
				state = 4;
			}
			break;
		case 4:
			MIXER_valveControl(SV4, true);
			state = 5;
			break;
		case 5:
			if (MIXER_tankIsEmpty(Tank1) && MIXER_tankIsEmpty(Tank2)
					&& MIXER_tankIsEmpty(Tank3)) {
				state = 6;
			}
			break;
		case 6:
			MIXER_valveControl(SV4, false);
			state = 7;
			break;
		case 7:
			if (MIXER_runMixer(val * 1000)) {
				state = 8;
			}
			break;
		case 8:
			MIXER_valveControl(SV5, true);
			state = 9;
			wait = SYSTICK_millis();
			break;
		case 9:
			if ((SYSTICK_millis() - wait) >= 2000) {
				state = 10;
			}
			break;
		case 10:
			MIXER_valveControl(SV5, false);
			display_done();
			wait = SYSTICK_millis();
			state = 11;
			break;
		case 11:
			if ((SYSTICK_millis() - wait) >= 2000) {
				state = 0;
				val = -1;
			}
			break;

			//reset process
		case 12:
			display_reset();
			MIXER_valveControl(SV1, false);
			MIXER_valveControl(SV2, false);
			MIXER_valveControl(SV3, false);
			MIXER_valveControl(SV4, true);
			MIXER_valveControl(SV5, true);
			state = 13;
			break;
		case 13:
			if (MIXER_tankIsEmpty(Tank1) && MIXER_tankIsEmpty(Tank2)
					&& MIXER_tankIsEmpty(Tank3)) {
				state = 14;
			}
			break;
		case 14:
			MIXER_reset();
			state = 0;
			val = -1;
			break;

		default:
			state = 0;
			val = -1;
		}
#else
		while (!IsKeyPressed(SWITCH_PIN))
		;

		MIXER_fillTank1(H2_Level);
		MIXER_fillTank2(H4_Level);
		MIXER_fillTank3(H8_Level);

		MIXER_valveControl(SV4, true);
		MIXER_tankIsEmpty(Tank1);
		MIXER_tankIsEmpty(Tank2);
		MIXER_tankIsEmpty(Tank3);

		MIXER_valveControl(SV4, false);
		MIXER_runMixer(5000);
		MIXER_valveControl(SV5, true);
		uint32_t start2 = SYSTICK_millis();
		while ((SYSTICK_millis() - start2) < 2000)
		;
		MIXER_valveControl(SV5, false);

		for (;;) {
			i++;
		}
#endif
	}

	/* Never leave main */
	return 0;
}

static void init() {
	PORTA->PCR[SWITCH_PIN] = PORT_PCR_MUX(1);
	PORTA->PCR[RESET_PIN] = PORT_PCR_MUX(1);

	LCD_initialize();
	display_main(0);

	ADCInit();
	ADCCalibrate();
	ADCInit();
}

/* Return 1 if the switch on given pin is pressed, 0 if not pressed.
 * */
static inline int IsKeyPressed(int pin) {
	if ((PTA->PDIR & (1 << pin)) == 0)
		return 1;
	else
		return 0;
}

void ADCInit(void) {
	// Povolit hodinovy signal pro ADC
	SIM->SCGC6 |= SIM_SCGC6_ADC0_MASK;

	// Zakazeme preruseni, nastavime kanal 31 = A/D prevodnik vypnut, jinak by zapisem
	// doslo ke spusteni prevodu
	// Vybereme single-ended mode
	ADC0->SC1[0] = ADC_SC1_ADCH(31);

	// Vyber hodinoveho signalu, preddelicky a rozliseni
	// Clock pro ADC nastavime <= 4 MHz, coz je doporuceno pro kalibraci.
	// Pri max. CPU frekvenci 48 MHz je bus clock 24 MHz, pri delicce = 8
	// bude clock pro ADC 3 MHz
	ADC0->CFG1 = ADC_CFG1_ADICLK(0) /* ADICLK = 0 -> bus clock */
	| ADC_CFG1_ADIV(3) /* ADIV = 3 -> clock/8 */
	| ADC_CFG1_MODE(2); /* MODE = 2 -> rozliseni 10-bit */

	// Do ostatnich registru zapiseme vychozi hodnoty:
	// Vybereme sadu kanalu "a", vychozi nejdelsi cas prevodu (24 clocks)
	ADC0->CFG2 = 0;

	// Softwarove spousteni prevodu, vychozi reference
	ADC0->SC2 = 0;

	// Hardwarove prumerovani vypnuto
	ADC0->SC3 = 0; /* default values, no averaging */

}

/*
 ADCCalibrate
 Kalibrace ADC.
 Kod prevzat z ukazkoveho kodu pro FRDM-KL25Z.
 Pri chybe kalibrace vraci 1, pri uspechu vraci 0
 */
uint32_t ADCCalibrate(void) {
	unsigned short cal_var;

	ADC0->SC2 &= ~ADC_SC2_ADTRG_MASK; /* Enable Software Conversion Trigger for Calibration Process */
	ADC0->SC3 &= (~ADC_SC3_ADCO_MASK & ~ADC_SC3_AVGS_MASK); /* set single conversion, clear avgs bitfield for next writing */

	ADC0->SC3 |= ( ADC_SC3_AVGE_MASK | ADC_SC3_AVGS(32)); /* turn averaging ON and set desired value */

	ADC0->SC3 |= ADC_SC3_CAL_MASK; /* Start CAL */

	/* Wait calibration end */
	while ((ADC0->SC1[0] & ADC_SC1_COCO_MASK) == 0)
		;

	/* Check for Calibration fail error and return */
	if ((ADC0->SC3 & ADC_SC3_CALF_MASK) != 0)
		return 1;

	// Calculate plus-side calibration
	cal_var = 0;
	cal_var = ADC0->CLP0;
	cal_var += ADC0->CLP1;
	cal_var += ADC0->CLP2;
	cal_var += ADC0->CLP3;
	cal_var += ADC0->CLP4;
	cal_var += ADC0->CLPS;

	cal_var = cal_var / 2;
	cal_var |= 0x8000; // Set MSB
	ADC0->PG = ADC_PG_PG(cal_var);

	// Calculate minus-side calibration
	cal_var = 0;
	cal_var = ADC0->CLM0;
	cal_var += ADC0->CLM1;
	cal_var += ADC0->CLM2;
	cal_var += ADC0->CLM3;
	cal_var += ADC0->CLM4;
	cal_var += ADC0->CLMS;

	cal_var = cal_var / 2;
	cal_var |= 0x8000; // Set MSB
	ADC0->MG = ADC_MG_MG(cal_var);

	ADC0->SC3 &= ~ADC_SC3_CAL_MASK;

	return 0;
}

static uint16_t read_analog_value(int min, int max) {
	// Spusteni prevodu na kanalu 11.
	// Protoze ostatni nastaveni v registru SC1 mame na 0, muzeme si dovolit
	// primo v nem prepsat hodnotu cislem kanalu. Lepsi reseni by bylo
	// "namaskovat" cislo kanalu bez zmeny hodnoty registru.
	ADC0->SC1[0] = ADC_SC1_ADCH(11);

	// Cekame na dokonceni prevodu
	while ((ADC0->SC1[0] & ADC_SC1_COCO_MASK) == 0)
		;

	// Ulozime vysledek prevodu
	return ADC0->R[0] / 1023.0 * (max - min) + min;
}

static void display_main(uint16_t mixing_time) {
	LCD_clear();
	LCD_set_cursor(1, 1);
	LCD_puts("Kavovar");
	LCD_set_cursor(2, 1);
	LCD_puts("SW1 - start");
	LCD_set_cursor(3, 1);
	LCD_puts("SW2 - reset");
	LCD_set_cursor(4, 1);
	static char buffer[64];
	snprintf(buffer, sizeof(buffer), "Cas mixovani: %d s", (int)mixing_time);
	LCD_puts(buffer);
}

static void display_run() {
	LCD_clear();
	LCD_set_cursor(1, 1);
	LCD_puts("Kava se pripravuje");
}

static void display_done() {
	LCD_clear();
	LCD_set_cursor(1, 1);
	LCD_puts("Kava je pripravena");
}

static void display_reset() {
	LCD_clear();
	LCD_set_cursor(1, 1);
	LCD_puts("Reset");
}

////////////////////////////////////////////////////////////////////////////////
// EOF
////////////////////////////////////////////////////////////////////////////////
