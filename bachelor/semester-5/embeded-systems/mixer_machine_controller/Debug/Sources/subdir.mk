################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Sources/drv_lcd.c \
../Sources/drv_systick.c \
../Sources/main.c \
../Sources/mixer.c 

OBJS += \
./Sources/drv_lcd.o \
./Sources/drv_systick.o \
./Sources/main.o \
./Sources/mixer.o 

C_DEPS += \
./Sources/drv_lcd.d \
./Sources/drv_systick.d \
./Sources/main.d \
./Sources/mixer.d 


# Each subdirectory must supply rules for building sources it contributes
Sources/%.o: ../Sources/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross ARM GNU C Compiler'
	arm-none-eabi-gcc -mcpu=cortex-m0plus -mthumb -O0 -fmessage-length=0 -fsigned-char -ffunction-sections -fdata-sections  -g3 -I"../Sources" -I"../Includes" -std=c99 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


