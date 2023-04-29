/*
 * mixer.cpp
 *
 *  Created on: 26. 10. 2021
 *      Author: m1_krcma
 */

#include "mixer.h"


//for async funcs
static volatile bool filling_1 = false;
static volatile bool filling_2 = false;
static volatile bool filling_3 = false;
static uint32_t mixer_start_time;
static volatile bool mixer_running = false;


static uint8_t getSVpin(SV_X sv);
static inline GPIO_Type * getHXport(uint8_t hx_pin);

void MIXER_init() {
	SYSTICK_initialize();

	// Enable clock for ports A, B, C, D, E
	SIM->SCGC5 |=
			(SIM_SCGC5_PORTA_MASK | SIM_SCGC5_PORTB_MASK | SIM_SCGC5_PORTC_MASK
					| SIM_SCGC5_PORTD_MASK | SIM_SCGC5_PORTE_MASK);

	// Set pin function to GPIO
	PORTE->PCR[SV1_PIN] = PORT_PCR_MUX(1);
	PORTD->PCR[H1_PIN] = PORT_PCR_MUX(1);
	PORTC->PCR[H2_PIN] = PORT_PCR_MUX(1);
	PORTD->PCR[H3_PIN] = PORT_PCR_MUX(1);

	PORTC->PCR[SV2_PIN] = PORT_PCR_MUX(1);
	PORTC->PCR[H4_PIN] = PORT_PCR_MUX(1);
	PORTD->PCR[H5_PIN] = PORT_PCR_MUX(1);

	PORTE->PCR[SV3_PIN] = PORT_PCR_MUX(1);
	PORTC->PCR[H6_PIN] = PORT_PCR_MUX(1);
	PORTD->PCR[H7_PIN] = PORT_PCR_MUX(1);
	PORTC->PCR[H8_PIN] = PORT_PCR_MUX(1);

	PORTE->PCR[SV4_PIN] = PORT_PCR_MUX(1);
	PORTE->PCR[SV5_PIN] = PORT_PCR_MUX(1);
	PORTD->PCR[MIXER_PIN] = PORT_PCR_MUX(1);

	// Set pin direction TANK1
	PTE->PDDR |= (1 << SV1_PIN);
	PTD->PDDR &= ~(1 << H1_PIN);
	PTC->PDDR &= ~(1 << H2_PIN);
	PTD->PDDR &= ~(1 << H3_PIN);

	// Set pin direction TANK2
	PTC->PDDR |= (1 << SV2_PIN);
	PTC->PDDR &= ~(1 << H4_PIN);
	PTD->PDDR &= ~(1 << H5_PIN);

	// Set pin direction TANK3
	PTE->PDDR |= (1 << SV3_PIN);
	PTC->PDDR &= ~(1 << H6_PIN);
	PTD->PDDR &= ~(1 << H7_PIN);
	PTC->PDDR &= ~(1 << H8_PIN);

	//MIX
	PTE->PDDR |= (1 << SV4_PIN);
	PTE->PDDR |= (1 << SV5_PIN);
	PTD->PDDR |= (1 << MIXER_PIN);
}

bool MIXER_fillTank1(Tank1_Levels stopLevel) {
	uint8_t HX;
	switch (stopLevel) {
	case H1_Level:
		HX = H1_PIN;
		break;
	case H2_Level:
		HX = H2_PIN;
		break;
	case H3_Level:
		HX = H3_PIN;
		break;
	default:
		return false;
	}
#ifdef ASYNCHRONOUS
	if (!filling_1) {
		MIXER_valveControl(SV1, true);
		filling_1 = true;
	}
	if ((getHXport(stopLevel)->PDIR & (1 << HX)) != 0) {
		MIXER_valveControl(SV1, false);
		filling_1 = false;
		return true;
	} else {
		return false;
	}
#else
	MIXER_valveControl(SV1, true);

	while ((getHXport(stopLevel)->PDIR & (1 << HX)) == 0)
	;
	MIXER_valveControl(SV1, false);
	return true;
#endif
}

bool MIXER_fillTank2(Tank2_Levels stopLevel) {
	uint8_t HX;
	switch (stopLevel) {
	case H4_Level:
		HX = H4_PIN;
		break;
	case H5_Level:
		HX = H5_PIN;
		break;
	default:
		return false;
	}
#ifdef ASYNCHRONOUS
	if (!filling_2) {
		MIXER_valveControl(SV2, true);
		filling_2 = true;
	}
	if ((getHXport(stopLevel)->PDIR & (1 << HX)) != 0) {
		MIXER_valveControl(SV2, false);
		filling_2 = false;
		return true;
	} else {
		return false;
	}
#else
	MIXER_valveControl(SV2, true);

	while ((getHXport(stopLevel)->PDIR & (1 << HX)) == 0)
	;
	MIXER_valveControl(SV2, false);

	return true;
#endif
}

bool MIXER_fillTank3(Tank3_Levels stopLevel) {
	uint8_t HX;
	switch (stopLevel) {
	case H6_Level:
		HX = H6_PIN;
		break;
	case H7_Level:
		HX = H7_PIN;
		break;
	case H8_Level:
		HX = H8_PIN;
		break;
	default:
		return false;
	}
#ifdef ASYNCHRONOUS
	if (!filling_3) {
		MIXER_valveControl(SV3, true);
		filling_3 = true;
	}
	if ((getHXport(stopLevel)->PDIR & (1 << HX)) != 0) {
		MIXER_valveControl(SV3, false);
		filling_3 = false;
		return true;
	} else {
		return false;
	}
#else
	MIXER_valveControl(SV3, true);

	while ((getHXport(stopLevel)->PDIR & (1 << HX)) == 0)
	;
	MIXER_valveControl(SV3, false);

	return true;
#endif
}

bool MIXER_tankIsEmpty(Tank t) {
#ifdef ASYNCHRONOUS
	switch (t) {
	case Tank1:
		if ((PTD->PDIR & (1 << H3_PIN)) == 0) {
			return true;
		}
		break;
	case Tank2:
		if ((PTD->PDIR & (1 << H5_PIN)) == 0) {
			return true;
		}
		break;
	case Tank3:
		if ((PTC->PDIR & (1 << H8_PIN)) == 0) {
			return true;
		}
		break;
	}
	return false;
#else
	switch (t) {
		case Tank1:
		while ((PTD->PDIR & (1 << H3_PIN)) != 0)
		;
		break;
		case Tank2:
		while ((PTD->PDIR & (1 << H5_PIN)) != 0)
		;
		break;
		case Tank3:
		while ((PTC->PDIR & (1 << H8_PIN)) != 0)
		;
		break;
	}
	return true;
#endif
}

void MIXER_valveControl(SV_X valve, bool open) {
	if (valve == SV2) {
		if (open) {
			PTC->PSOR = (1 << getSVpin(valve));
		} else {
			PTC->PCOR = (1 << getSVpin(valve));
		}
	} else {
		if (open) {
			PTE->PSOR = (1 << getSVpin(valve));
		} else {
			PTE->PCOR = (1 << getSVpin(valve));
		}
	}
}

bool MIXER_runMixer(uint32_t time) {
#ifdef ASYNCHRONOUS
	if(!mixer_running) {
		mixer_start_time = SYSTICK_millis();
		PTD->PSOR = (1 << MIXER_PIN);
		mixer_running = true;
	}

	if (SYSTICK_millis() - mixer_start_time >= time) {
		PTD->PCOR = (1 << MIXER_PIN);
	} else {
		return false;
	}
#else
	mixer_start_time = SYSTICK_millis();
	PTD->PSOR = (1 << MIXER_PIN);

	while (SYSTICK_millis() - mixer_start_time < time)
	;

	PTD->PCOR = (1 << MIXER_PIN);
	return true;
#endif
}

void MIXER_reset() {
	filling_1 = false;
	filling_2 = false;
	filling_3 = false;
	mixer_running = false;

	MIXER_valveControl(SV1, false);
	MIXER_valveControl(SV2, false);
	MIXER_valveControl(SV3, false);
	MIXER_valveControl(SV4, false);
	MIXER_valveControl(SV5, false);
}

static inline GPIO_Type * getHXport(uint8_t hx_pin_number) {
	return hx_pin_number % 2 == 0 ? PTD : PTC;
}

static uint8_t getSVpin(SV_X sv) {
	switch (sv) {
	case SV1:
		return SV1_PIN;
	case SV2:
		return SV2_PIN;
	case SV3:
		return SV3_PIN;
	case SV4:
		return SV4_PIN;
	case SV5:
		return SV5_PIN;
	}
}

