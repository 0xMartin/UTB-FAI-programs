/*
 * mixer.h
 *
 *  Created on: 26. 10. 2021
 *      Author: m1_krcma
 */

#ifndef SOURCES_MIXER_H_
#define SOURCES_MIXER_H_

#include "MKL25Z4.h"
#include "drv_systick.h"
#include <stdint.h>
#include <stdbool.h>


#define ASYNCHRONOUS


/**
 Pin MCU Funkce modelu
 E0 Ventil SV4
 E1 Ventil SV5
 E4 Ventil SV3
 E3 Nezapojeno
 E5 Ventil SV1
 C1 Ventil SV2
 B1 Nezapojeno
 D0 Míchadlo v mísicí nádrži
 C16 Hladinomìr H2
 D5 Hladinomìr H7
 D4 Hladinomìr H5
 D2 Hladinomìr H3
 3
 A5MPC – Programování mikropoèítaèù 21.10.2021
 D3 Hladinomìr H1
 C7 Hladinomìr H6
 C6 Hladinomìr H8
 C5 Hladinomìr H4
 */

#define SWITCH_PIN (4) // A4

#define H1_PIN 	(3)		//D3
#define H2_PIN 	(16)	//C16
#define H3_PIN 	(2)		//D2
#define H4_PIN 	(5)		//C5
#define H5_PIN 	(4)		//D4
#define H6_PIN 	(7)		//C7
#define H7_PIN 	(5)		//D5
#define H8_PIN 	(6)		//C6

#define SV1_PIN (5)		//E5
#define SV2_PIN (1)		//C1
#define SV3_PIN (4)		//E4
#define SV4_PIN (0)		//E0
#define SV5_PIN (1)		//E1

#define MIXER_PIN (0) 	//D0

typedef enum {
	SV1 = 0, SV2 = 1, SV3 = 2, SV4 = 3, SV5 = 4
} SV_X;

typedef enum {
	H1_Level = 0, H2_Level = 1, H3_Level = 2
} Tank1_Levels;

typedef enum {
	H4_Level = 3, H5_Level = 4
} Tank2_Levels;

typedef enum {
	H6_Level = 5, H7_Level = 6, H8_Level = 7
} Tank3_Levels;

typedef enum {
	Tank1, Tank2, Tank3
} Tank;

void MIXER_init();

bool MIXER_fillTank1(Tank1_Levels stopLevle);

bool MIXER_fillTank2(Tank2_Levels stopLevle);

bool MIXER_fillTank3(Tank3_Levels stopLevle);

bool MIXER_tankIsEmpty(Tank t);

void MIXER_valveControl(SV_X valve, bool open);

bool MIXER_runMixer(uint32_t time);

void MIXER_reset();

#endif /* SOURCES_MIXER_H_ */
