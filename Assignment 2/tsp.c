#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>
#include <string.h>

#define TASK_TAG 1
#define AVAILABLE_TAG 2
#define FINALIZE_TAG 3
#ifdef ENABLE_DEBUG
#define DEBUG(F,...) printf(F"\n",##__VA_ARGS__)
#else
#define DEBUG(F,...)
#endif

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


List *listNew()
{
    List *newList = malloc(sizeof(List));
    newList->head = NULL;
    newList->tail = NULL;
    newList->itemsCount = 0;
}

void listEnqueue(List * list, void *data)
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

    DEBUG("listEnqueue: data=%p", data);
}

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
    DEBUG("removeItem: list->head=%p, list->itemsCount=%d", list->head,
	  list->itemsCount);
    return data;

}


void *listDequeue(List * list)
{
    return removeItem(list, list->head);
}

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

void printMatrix(int *matrix, int n)
{
    int i, j;
    for (i = 0; i < n; i++) {
	for (j = 0; j < n; j++) {
	    printf("%d ", matrix[n * i + j]);
	}
	printf("\n");
    }
}

void printRoute(int *route, int routeLen)
{
    int i;
    for (i = 0; i < routeLen; i++) {
	printf("%d ", route[i]);
    }
    printf("]\n");
}

void printTask(TSP_task * task)
{
    printf
	("TASK:\n\tlongMin: %d\n\tpartialLong: %d\n\tvisitedCount: %d\n\tcityToVisit: %d\n\t[",
	 task->longMin, task->partialLong, task->visitedCount,
	 task->cityToVisit);
    printRoute(task->visited, task->visitedCount);
}

MPI_Datatype constructWorkType(int n)
{
    MPI_Datatype workType;
    MPI_Type_contiguous(n + 4, MPI_INT, &workType);
    MPI_Type_commit(&workType);

    return workType;
}

void hostWork(int *matrix, int n)
{
    MPI_Datatype taskType = constructWorkType(n);
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
	    listEnqueue(taskBag, task);
    }

    do {
	DEBUG("available Workers: %d, taskBag: %d", availableWorkers->itemsCount, taskBag->itemsCount);
	DEBUG("[Master] Waiting for messages..");
	MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	DEBUG("[Master] New message from rank=%d and tag=%d", status.MPI_SOURCE, status.MPI_TAG);
	if (status.MPI_TAG == TASK_TAG) {
	    MPI_Get_count(&status, taskType, &elementsCount);
	    tasks = newTask(elementsCount, n);
	    MPI_Recv(tasks, elementsCount, taskType, status.MPI_SOURCE,
		     status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    DEBUG("[Master] Message with %d tasks", elementsCount);
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
		    listEnqueue(taskBag, t);
		}
	    }
	    free(tasks);
	    // Assinging tasks
	    while (availableWorkers->itemsCount > 0 && taskBag->itemsCount > 0) {
		    task = listDequeue(taskBag);
		    rank = (int) listDequeue(availableWorkers);
		    task->longMin = longMin;
		    MPI_Send(task, 1, taskType, rank, TASK_TAG, MPI_COMM_WORLD);
		    free(task);
	    }
	} else {
	    // AVAILABLE_TAG
	    MPI_Recv(&rank, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    if (taskBag->itemsCount > 0) {
		    task = listDequeue(taskBag);
		    task->longMin = longMin;
		    MPI_Send(task, 1, taskType, rank, TASK_TAG, MPI_COMM_WORLD);
		    free(task);
	    } else {
		    listEnqueue(availableWorkers, (void *) rank);
	    }
	}
    }
    while (availableWorkers->itemsCount < 3 || taskBag->itemsCount > 0);
    
    // Sending message of completion
    while ((rank = (int) listDequeue(availableWorkers)) != 0)
	MPI_Send(&rank, 1, MPI_INT, rank, FINALIZE_TAG, MPI_COMM_WORLD);

    // printing the best route
    printf("Best Route:\n\t[ ");
    bestRoute[n] = 0;
    printRoute(bestRoute, n + 1);
    printf("Cost of the route: %d\n", longMin);
}

void findCitiesToVisit(int *visited, int visitedCount, int *ret, int n)
{
    int i, j;
    int sortedArr[n];

    memset(sortedArr, 0, sizeof(sortedArr));
    for (i = 0; i < visitedCount; i++) {
	DEBUG("find Cities To Visit: visited[i=%d]=%d", i, visited[i]);
	sortedArr[visited[i]] = 1;
    }
    for (i = 0, j = 0; i < n; i++) {
	if (sortedArr[i] == 0)
	    ret[j++] = i;
    }
}

void distributedWork(int *matrix, int n, int rank)
{
    MPI_Datatype taskType = constructWorkType(n);
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

	MPI_Recv(task, 1, taskType, 0, TASK_TAG, MPI_COMM_WORLD,
		 MPI_STATUS_IGNORE);
	DEBUG("Tasks recieved from rank=%d", rank);
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
		    DEBUG("task->visitedCount=%d", task->visitedCount);
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


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // refering process with rank 0 as our host process
    if (rank == 0) {
	    n = atoi(argv[1]);
	    matrix = readMatrix(argv[2], n);
	    MPI_Type_contiguous(n * n, MPI_INT, &matrixType);
	    MPI_Type_commit(&matrixType);

	    //Sending integer n from host
	    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	    //Sending matrix from host
	    MPI_Bcast(matrix, 1, matrixType, 0, MPI_COMM_WORLD);
	    hostWork(matrix, n);

    } else {
	//Reciving n from host process       
	    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	    MPI_Type_contiguous(n * n, MPI_INT, &matrixType);
	    MPI_Type_commit(&matrixType);
	    matrix = malloc(n * n * sizeof(int));

	    //Reciving matrix
	    MPI_Bcast(matrix, 1, matrixType, 0, MPI_COMM_WORLD);
	    distributedWork(matrix, n, rank);
    }

    MPI_Finalize();
    return 0;
}
