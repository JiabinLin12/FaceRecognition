#pragma once
#include <syslog.h>
#include <stdio.h>

//#define LOG_HELPER(fmt,...) printf(fmt "\n%s",__VA_ARGS__)
//#define LOG(...) LOG_HELPER(__VA_ARGS__,"")


#define VERBOSE 1
#define VERYVERBOSE 0
#define LOG(level, message, ...) {if(VERBOSE) syslog(level,message, ##__VA_ARGS__);\
								if(VERYVERBOSE) printf(message "\n",##__VA_ARGS__);}
