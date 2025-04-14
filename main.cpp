#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

float tdiff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec-start->tv_sec) + 1e-6*(end->tv_usec-start->tv_usec);
}

struct Planet {
   double mass;
   double x;
   double y;
   double vx;
   double vy;
};

unsigned long long seed = 100;

unsigned long long randomU64() {
  seed ^= (seed << 21);
  seed ^= (seed >> 35);
  seed ^= (seed << 4);
  return seed;
}

double randomDouble() {
    unsigned long long a = randomU64() >> (64 - 26);
    unsigned long long b = randomU64() >> (64 - 26);
    return ((a << 27) + b) / (double)(1LL << 53);
}

typedef struct QuadNode {
    double xmid;
    double ymid;
    double half_size;
    double mass;
    double cm_x;
    double cm_y;
    int bodyIndex;
    struct QuadNode* NW;
    struct QuadNode* NE;
    struct QuadNode* SW;
    struct QuadNode* SE;
} QuadNode;

static unsigned long long seed = 100ULL;
unsigned long long randomU64() {
    seed ^= (seed << 21);
    seed ^= (seed >> 35);
    seed ^= (seed << 4);
    return seed;
}


double randomDouble() {
    unsigned long long a = randomU64() >> (64 - 26);
    unsigned long long b = randomU64() >> (64 - 26);
    return ((a << 27) + b) / (double)(1ULL << 53);
}
float tdiff(struct timeval* start, struct timeval* end) {
    return (end->tv_sec - start->tv_sec) + 1e-6f * (end->tv_usec - start->tv_usec);
}

QuadNode* createNode(double xmid, double ymid, double half_size) {
    QuadNode* node = (QuadNode*)malloc(sizeof(QuadNode));
    node->xmid = xmid;
    node->ymid = ymid;
    node->half_size = half_size;
    node->mass = 0.0;
    node->cm_x = 0.0;
    node->cm_y = 0.0;
    node->bodyIndex = -1;
    node->NW = node->NE = node->SW = node->SE = NULL;
    return node;
}

void insertBody(QuadNode* node, int i, Planet* bodies) {
    if (node->bodyIndex == -1 && node->NW == NULL) {
        node->bodyIndex = i;
        node->mass = bodies[i].mass;
        node->cm_x = bodies[i].x;
        node->cm_y = bodies[i].y;
        return;
    }
    double totalMass = node->mass + bodies[i].mass;
    node->cm_x = (node->cm_x * node->mass + bodies[i].x * bodies[i].mass) / totalMass;
    node->cm_y = (node->cm_y * node->mass + bodies[i].y * bodies[i].mass) / totalMass;
    node->mass = totalMass;
    if (node->NW == NULL) {
        double hs = node->half_size / 2.0;
        node->NW = createNode(node->xmid - hs, node->ymid + hs, hs);
        node->NE = createNode(node->xmid + hs, node->ymid + hs, hs);
        node->SW = createNode(node->xmid - hs, node->ymid - hs, hs);
        node->SE = createNode(node->xmid + hs, node->ymid - hs, hs);
        int oldIndex = node->bodyIndex;
        node->bodyIndex = -1;
        if (bodies[oldIndex].x <= node->xmid) {
            if (bodies[oldIndex].y >= node->ymid) insertBody(node->NW, oldIndex, bodies);
            else insertBody(node->SW, oldIndex, bodies);
        } else {
            if (bodies[oldIndex].y >= node->ymid) insertBody(node->NE, oldIndex, bodies);
            else insertBody(node->SE, oldIndex, bodies);
        }
    }
    if (bodies[i].x <= node->xmid) {
        if (bodies[i].y >= node->ymid) insertBody(node->NW, i, bodies);
        else insertBody(node->SW, i, bodies);
    } else {
        if (bodies[i].y >= node->ymid) insertBody(node->NE, i, bodies);
        else insertBody(node->SE, i, bodies);
    }
}

void computeForceBarnesHut(QuadNode* node, Planet* bodies, int i,
                           double theta, double G, double softening,
                           double* fx, double* fy)
{
    if (!node || node->mass <= 0.0) return;
    double dx = node->cm_x - bodies[i].x;
    double dy = node->cm_y - bodies[i].y;
    double dist = sqrt(dx*dx + dy*dy) + softening;
    if (node->NW == NULL) {
        if (node->bodyIndex == i || node->bodyIndex < 0) return;
        double F = G * bodies[i].mass * bodies[node->bodyIndex].mass / (dist*dist);
        *fx += F * (dx/dist);
        *fy += F * (dy/dist);
        return;
    }
    double s = node->half_size * 2.0;
    if (s / dist < theta) {
        double F = G * bodies[i].mass * node->mass / (dist*dist);
        *fx += F * (dx/dist);
        *fy += F * (dy/dist);
    } else {
        computeForceBarnesHut(node->NW, bodies, i, theta, G, softening, fx, fy);
        computeForceBarnesHut(node->NE, bodies, i, theta, G, softening, fx, fy);
        computeForceBarnesHut(node->SW, bodies, i, theta, G, softening, fx, fy);
        computeForceBarnesHut(node->SE, bodies, i, theta, G, softening, fx, fy);
    }
}

void freeTree(QuadNode* node) {
    if (!node) return;
    freeTree(node->NW);
    freeTree(node->NE);
    freeTree(node->SW);
    freeTree(node->SE);
    free(node);
}

void computeForceDirect(Planet* bodies, int n, int i, double G, double softening, double* fx, double* fy) {
    double xi = bodies[i].x;
    double yi = bodies[i].y;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double dx = bodies[j].x - xi;
        double dy = bodies[j].y - yi;
        double distSqr = dx*dx + dy*dy + softening;
        double dist = sqrt(distSqr);
        double F = G * bodies[i].mass * bodies[j].mass / distSqr;
        *fx += F * (dx/dist);
        *fy += F * (dy/dist);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <nplanets> <timesteps>\n", argv[0]);
        return 1;
    }
    int nplanets = atoi(argv[1]);
    int timesteps = atoi(argv[2]);
    double dt = 0.01;
    double G = 6.6743;
    double softening = 1e-4;
    Planet* planets = (Planet*)malloc(sizeof(Planet)*nplanets);
    for (int i = 0; i < nplanets; i++) {
        planets[i].mass = randomDouble() + 0.1;
        planets[i].x = randomDouble() * 100.0 - 50.0;
        planets[i].y = randomDouble() * 100.0 - 50.0;
        planets[i].vx = randomDouble() * 5.0 - 2.5;
        planets[i].vy = randomDouble() * 5.0 - 2.5;
    }

    double theta = 0.5;     
    int SWITCH_STEP = 2000; 

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int step = 0; step < timesteps; step++) {
        if (step < SWITCH_STEP) {
            double minX = planets[0].x, maxX = planets[0].x;
            double minY = planets[0].y, maxY = planets[0].y;
            for (int i = 1; i < nplanets; i++) {
                if (planets[i].x < minX) minX = planets[i].x;
                if (planets[i].x > maxX) maxX = planets[i].x;
                if (planets[i].y < minY) minY = planets[i].y;
                if (planets[i].y > maxY) maxY = planets[i].y;
            }
            double cx = 0.5*(minX + maxX);
            double cy = 0.5*(minY + maxY);
            double maxSpan = (maxX - minX) > (maxY - minY) ? (maxX - minX) : (maxY - minY);
            QuadNode* root = createNode(cx, cy, 0.5*maxSpan+1e-6);

            for (int i = 0; i < nplanets; i++) {
                insertBody(root, i, planets);
            }

            #pragma omp parallel for
            for (int i = 0; i < nplanets; i++) {
                double fx = 0.0, fy = 0.0;
                computeForceBarnesHut(root, planets, i, theta, G, softening, &fx, &fy);
                double ax = fx / planets[i].mass;
                double ay = fy / planets[i].mass;
                planets[i].vx += dt * ax;
                planets[i].vy += dt * ay;
            }
            #pragma omp parallel for
            for (int i = 0; i < nplanets; i++) {
                planets[i].x += dt * planets[i].vx;
                planets[i].y += dt * planets[i].vy;
            }
            freeTree(root);
        } else {
            #pragma omp parallel for
            for (int i = 0; i < nplanets; i++) {
                double fx = 0.0, fy = 0.0;
                computeForceDirect(planets, nplanets, i, G, softening, &fx, &fy);
                double ax = fx / planets[i].mass;
                double ay = fy / planets[i].mass;
                planets[i].vx += dt * ax;
                planets[i].vy += dt * ay;
            }
            #pragma omp parallel for
            for (int i = 0; i < nplanets; i++) {
                planets[i].x += dt * planets[i].vx;
                planets[i].y += dt * planets[i].vy;
            }
        }
        if (step % 1000 == 0) {
            printf("Step %d: last planet at (%.4f, %.4f)\n",
                   step, planets[nplanets-1].x, planets[nplanets-1].y);
        }
    }
    gettimeofday(&end, NULL);
    float runtime = tdiff(&start, &end);
    printf("Total time: %.6f s\n", runtime);
    printf("Final location (%.5f, %.5f)\n", planets[nplanets-1].x, planets[nplanets-1].y);

    free(planets);
    return 0;
}
