#ifndef UTIL_H
#define UTIL_H
#endif

#include <string.h>
#include <cstdlib>
#include <iostream>

using namespace std;

void usage( char *exe );

int process_command_line_option( int argc, char** argv, int* idx );
