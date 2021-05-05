#include <math.h>
#include <stdio.h>
#define TRUE 1
#define FALSE 0
#define U32_T unsigned int
int completion_time_feasibility(U32_T numServices,U32_T period[], U32_T wcet[],U32_T deadline[])
{
    int i, j;
    U32_T an, anext;
    // assume feasible until we find otherwise
    int set_feasible=TRUE;
    //printf(“numServices=%d\n”, numServices);
    for (i=0; i < numServices; i++)
    {
        an=0; anext=0;
        for (j=0; j <= i; j++)
        {
            an+=wcet[j];
        }
        //printf(“i=%d, an=%d\n”, i, an);
        while(1)
        {
            anext=wcet[i];
            for (j=0; j < i; j++)
                anext += ceil(((double)an)/((double)period[j]))*wcet[j];
            if (anext == an)
                break;
            else
                an=anext; 
            //printf(“an=%d, anext=%d\n”, an, anext);
        }
            //printf(“an=%d, deadline[%d]=%d\n”, an, i, deadline[i]);
        if (an > deadline[i])
        {
            set_feasible=FALSE;
        }
    }
    return set_feasible;
}

int main(){
    printf("Test Input: C1 =13, C2 = 65, T1 = 50, T2 = 1000, T=D \n");
    U32_T numServices = 2;
    U32_T period[] = {50, 1000};
    U32_T wcet[] = {13,65};
    U32_T deadline[] = {50,1000};
    if (completion_time_feasibility(numServices,period,wcet,deadline))
        printf("THIS SET OF SERVICE IS SCHEDULEABLE");
    else
        printf("THIS SET OF SERVICE IS NOT SCHEDUABLE");
    return 0;
}