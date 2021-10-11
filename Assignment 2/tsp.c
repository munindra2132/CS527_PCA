/*
*
*		CS527 Parallel Computer Architecture
*		Assignemnt 2	Parallel Programming
*		Topic - MPI (Message Passing Interface)		
*	Solution of Travelling Salesman in C using MPI 
*         
*		Group - 2
*   Members -  Abhishek Kumar 180101003
*			   Aniraj Kumar	  180101007
*			   Munindra Naik  180101045
*			   Ritik Mandloi 180101066
*
*	Approach - The idea is to have a bag of tasks, in which initially a single task is placed globally. Multiple processes take tasks from the bag and process them, often generating 
*			   new tasks that will be corresponding to subproblems. The new  tasks are placed in the bag. The computation ends when there does not exist any tasks to perform. 
*			   We will have two types of processes one will be the host and worker, there will be one host process which will assign tasks to the worker process and the worker will be performing tasks 
*              and returning either a solution containing the hamiltonian path or further subproblems which need to be solved. The host 
*              process will maintain a variable which will store the answer of the problem which will be updated whenever a worker process finds  a solution and 
*              compares if a  better value is found or not. 
*			
*	
*	Compilation and Execution 
*   1) mpicc -w tsp.c -o tsp
*	2) mpirun --np 'number of processes' ./tsp 'number of cities' 'filename'
*                         or 
*	   mpirun --np 'number of processes' ./tsp - after that enter number of cities and then a 2d matrix where (i,j) refers to distance b/w ith and jth city
*												 if there is no path b/w ith and jth then enter '-1' and for (i,i) entry enter '0'
*/
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <mpi.h>

//defining a variable which signifies a significantly long path if there is no edge b/w 
#define INF 2e9+7
// defining tags which occurs in messages
#define TASK_TAG 1
#define AVAILABLE_TAG 2
#define FINALIZE_TAG 3


typedef struct {
    int longMin;
    int partialLong;
    int cityToVisit;
    int visitedCount;
    int visited[];
} TSP_task;

typedef struct _ListItem {
    void *data;
    struct _ListItem *prev, *next;
} ListItem;

typedef struct {
    ListItem *head;
    ListItem *tail;
    int itemsCount;
} List;


//creating a data structure list which stores pool of tasks and available workers
List *listNew()
{
    List *newList = malloc(sizeof(List));
    newList->head = NULL;
    newList->tail = NULL;
    newList->itemsCount = 0;
}

// method to add item to list
void enqueue(List * list, void *data)
{
    ListItem *item;
    item = malloc(sizeof(ListItem));
    item->data = data;
    item->prev = list->tail;
    list->tail = item;

    item->next = NULL;
    if (item->prev)
	item->prev->next = item;

    if (list->head == NULL) {
	list->head = item;
    }
    list->itemsCount++;
}

// helper method to remove item form list
void *removeItem(List * list, ListItem * item)
{
    void *data;
    if (item == NULL)
	return NULL;

    if (list->head == item) {
	list->head = item->next;
    }

    if (list->tail == item) {
	list->tail = item->prev;
    }

    if (item->next) {
	item->next->prev = item->prev;
    }

    if (item->prev) {
	item->prev->next = item->next;
    }

    data = item->data;
    list->itemsCount--;
    free(item);
    return data;

}

// method to remove item from list
void *dequeue(List * list)
{
    return removeItem(list, list->head);
}

//function to create new task
TSP_task *newTask(int m, int n)
{
    TSP_task *task;
    int i, j;
    task = malloc((sizeof(TSP_task) + sizeof(int) * n) * m);

    for (j = 0; j < m; j++) {
	for (i = 0; i < n; i++) {
	    task[j].visited[i] = 0;
	}
    }

    return task;
}

TSP_task *calc_task_pointer(void *base, int n, int idx)
{
    return base + (sizeof(TSP_task) + n * sizeof(int)) * idx;
}

//function to read the input if given filename
int *readMatrix(char *fileName, int n)
{

    int i;
    int *matrix = calloc(n * n, sizeof(int));
    FILE *fp = fopen(fileName, "r");
    for (i = 0; i < n * n; i++)
	fscanf(fp, "%d", &matrix[i]);
    fclose(fp);
    return matrix;
}

//function to print the route after finding the solution
void printRoute(int *route, int routeLen)
{
    int i;
    for (i = 0; i < routeLen; i++) {
	printf("%d ", route[i]);
    }
    printf("]\n");
}

// defining the work that has to be done by host process
void hostWork(int *matrix, int n)
{
    MPI_Datatype taskType; 
	MPI_Type_contiguous(n + 4, MPI_INT, &taskType);
    MPI_Type_commit(&taskType);
    
	MPI_Status status;
    TSP_task *task, *tasks, *t;
    List *taskBag = listNew();
    List *availableWorkers = listNew();
    int elementsCount;
    int i;
    int longMin = INT_MAX;
    int *bestRoute = calloc(n + 1, sizeof(int));
    int rank;

    // adds the initial tasks
    for (i = 0; i < n - 1; i++) {
	    task = newTask(1, n);
	    task->longMin = INT_MAX;
	    task->cityToVisit = i + 1;
	    task->visitedCount = 1;
	    task->visited[0] = 0;
	    task->partialLong = 0;
	    enqueue(taskBag, task);
    }

    do {
	MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	if (status.MPI_TAG == TASK_TAG) {
	    MPI_Get_count(&status, taskType, &elementsCount);
	    tasks = newTask(elementsCount, n);
	    MPI_Recv(tasks, elementsCount, taskType, status.MPI_SOURCE,
		     status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    for (i = 0; i < elementsCount; i++) {
		task = calc_task_pointer(tasks, n, i);

		//checking if walked the way to start
		if (task->visitedCount > n) {
		    if (longMin > task->partialLong) {
			longMin = task->partialLong;
			memcpy(bestRoute, task->visited, sizeof(int) * n);
		    }
		} else {
		    t = newTask(1, n);
		    memcpy(t, task, sizeof(TSP_task) + sizeof(int) * n);
		    enqueue(taskBag, t);
		}
	    }
	    free(tasks);
	    // Assinging tasks
	    while (availableWorkers->itemsCount > 0 && taskBag->itemsCount > 0) {
		    task = dequeue(taskBag);
		    rank = (int) dequeue(availableWorkers);
		    task->longMin = longMin;
		    MPI_Send(task, 1, taskType, rank, TASK_TAG, MPI_COMM_WORLD);
		    free(task);
	    }
	} else {
	    // AVAILABLE_TAG
	    MPI_Recv(&rank, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    if (taskBag->itemsCount > 0) {
		    task = dequeue(taskBag);
		    task->longMin = longMin;
		    MPI_Send(task, 1, taskType, rank, TASK_TAG, MPI_COMM_WORLD);
		    free(task);
	    } else {
		    enqueue(availableWorkers, (void *) rank);
	    }
	}
    }
    while (availableWorkers->itemsCount < 3 || taskBag->itemsCount > 0);
    
    // Sending message of completion
    while ((rank = (int) dequeue(availableWorkers)) != 0)
	MPI_Send(&rank, 1, MPI_INT, rank, FINALIZE_TAG, MPI_COMM_WORLD);

    // printing the best route
    printf("Best Route:\n\t[ ");
    bestRoute[n] = 0;
    printRoute(bestRoute, n + 1);
    printf("Cost of the route: %d\n", longMin);
}

//performing the allotment of tasks to host process
void findCitiesToVisit(int *visited, int visitedCount, int *ret, int n)
{
    int i, j;
    int sortedArr[n];

    memset(sortedArr, 0, sizeof(sortedArr));
    for (i = 0; i < visitedCount; i++) {
		sortedArr[visited[i]] = 1;
    }
    for (i = 0, j = 0; i < n; i++) {
	if (sortedArr[i] == 0)
	    ret[j++] = i;
    }
}

//performing the work given by host process
void distributedWork(int *matrix, int n, int rank)
{
    MPI_Datatype taskType; 
	MPI_Type_contiguous(n + 4, MPI_INT, &taskType);
    MPI_Type_commit(&taskType);
    
	TSP_task *task, *t, *tasks;
    MPI_Status status;
    int citiesToVisit;
    int cities[n];
    int i;

    task = newTask(1, n);
    do {
	MPI_Send(&rank, 1, MPI_INT, 0, AVAILABLE_TAG, MPI_COMM_WORLD);

	MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	if (status.MPI_TAG == FINALIZE_TAG)
	    break;

	MPI_Recv(task, 1, taskType, 0, TASK_TAG, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	if (task->visitedCount >= n) {
	    // visited all of the cities, going back to 0
	    task->partialLong += matrix[task->visited[task->visitedCount - 1] * n];
	    task->visitedCount++;
	    // Pruning : if partialLong exceeds or equals longMin, not sent back
	    if (task->partialLong < task->longMin)
		MPI_Send(task, 1, taskType, 0, TASK_TAG, MPI_COMM_WORLD);
	} else {
	    task->partialLong +=matrix[task->visited[task->visitedCount - 1] * n + task->cityToVisit];
	    task->visited[task->visitedCount] = task->cityToVisit;
	    task->visitedCount++;
	    if (task->partialLong < task->longMin) {
		citiesToVisit = n - task->visitedCount;
		if (citiesToVisit > 0) {
		    tasks = newTask(citiesToVisit, n);
		    findCitiesToVisit(task->visited, task->visitedCount,cities, n);
		    for (i = 0; i < citiesToVisit; i++) {
				t = calc_task_pointer(tasks, n, i);
				memcpy(t, task,sizeof(TSP_task) + sizeof(int) * n);
				t->cityToVisit = cities[i];
		    }
		    MPI_Send(tasks, citiesToVisit, taskType, 0, TASK_TAG,MPI_COMM_WORLD);
		    free(tasks);
		} else {
		    MPI_Send(task, 1, taskType, 0, TASK_TAG,MPI_COMM_WORLD);
		}
	    }
	}
    }
    while (1);
}


int main(int argc, char *argv[])
{
    int rank, size;
    int *matrix;		//buffer
    int n;

    MPI_Datatype matrixType;

	//initializing the MPI execution
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // refering process with rank 0 as our host process
    if (rank == 0) {
		if(argc == 3){
	    	n = atoi(argv[1]);
	    	matrix = readMatrix(argv[2], n);
		}else{
			scanf("%d",&n);
			matrix = calloc(n*n,sizeof(int));
			for(int i=0;i<n;i++){
				for(int j=0;j<n;j++){
					scanf("%d",&matrix[n*i+j]);
					if(matrix[n*i+j] == -1) matrix[n*i+j] = INF;
					printf("%d ",matrix[n*i+j]);
				}
				printf("\n");
			} 
		}
	    MPI_Type_contiguous(n * n, MPI_INT, &matrixType);
	    MPI_Type_commit(&matrixType);

	    //broadcasting the number of cities from host
	    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	    //broadcasting the distance matrix from host
	    MPI_Bcast(matrix, 1, matrixType, 0, MPI_COMM_WORLD);

	    //performing the allotment of tasks to host process
		hostWork(matrix, n);

    } else {
		//Reciving the value of number of cities from host process       
	    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	    MPI_Type_contiguous(n * n, MPI_INT, &matrixType);
	    MPI_Type_commit(&matrixType);
	    matrix = malloc(n * n * sizeof(int));

	    //Reciving the distance matrix from host
	    MPI_Bcast(matrix, 1, matrixType, 0, MPI_COMM_WORLD);

		//performing the work given by host process
	    distributedWork(matrix, n, rank);
    }

	//shutting down the mpi execution
    MPI_Finalize();
    return 0;
}
