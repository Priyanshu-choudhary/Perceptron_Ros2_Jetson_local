// Machine-speed probe: every 2 s time a fixed compute kernel (~3 ms) and log
// "epoch_s  ms". Uses <0.2% CPU. Rising ms = the machine got slower.
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(void){
  setvbuf(stdout,NULL,_IOLBF,0);
  volatile double sink=0;
  for(;;){
    double best=1e9;
    for(int r=0;r<3;r++){
      double t0=now(),a=0;
      for(int i=0;i<200000;i++){a+=sin(i*0.001)*cos(a*1e-9+i);}
      sink+=a; double d=now()-t0; if(d<best)best=d;
    }
    printf("%.3f %.4f\n",(double)time(NULL),best*1000);
    sleep(2);
  }
}
