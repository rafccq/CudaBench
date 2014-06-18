#pragma once

#include <biggle.h>
#include <GL/glfw.h>

#define N_KEYS 256 + 69

const char* get_key_name(int key);

const char* get_action_name(int action);

const char* get_button_name(int button);

const char* get_character_string(int character);

void GLFWCALL window_size_callback(int width, int height);

int GLFWCALL window_close_callback(void);

void GLFWCALL window_refresh_callback(void);

void GLFWCALL mouse_button_callback(int button, int action);

void GLFWCALL mouse_position_callback(int x, int y);

void GLFWCALL mouse_wheel_callback(int position);

void GLFWCALL key_callback(int key, int action);

void GLFWCALL char_callback(int character, int action);
