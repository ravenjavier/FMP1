// MACHINE PROBLEM 01 - Campus Traffic Signal Visualizer
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
using namespace std;

namespace {

const int   NUM_LIGHTS    = 3;       // Red, Yellow, Green (top to bottom)
const int   DISC_SEGMENTS = 48;      // triangles per circular indicator
const float PI            = 3.14159265f;

const float LIGHT_RADIUS   = 0.17f;
const float LIGHT_GAP      = 0.08f;  // space between neighbouring indicators
const float HOUSING_PAD    = 0.10f;  // inner margin of the housing
const float HOUSING_HALF_W = 0.28f;
const float HOUSING_BOTTOM = -0.50f;
const float FRAME_BORDER   = 0.025f;
const float HALO_SCALE     = 1.40f;  // glow radius relative to the indicator
const float DIM_FACTOR     = 0.16f;  // brightness of an inactive indicator

struct Range { GLint first; GLsizei count; };   // slice of the VBO
struct Color { float r, g, b; };

// Index 0 = top light. Extend these tables if NUM_LIGHTS is increased.
const Color LIGHT_ON[NUM_LIGHTS] = {
    {0.96f, 0.12f, 0.12f},   // red
    {1.00f, 0.82f, 0.10f},   // yellow
    {0.12f, 0.88f, 0.35f}    // green
};

enum Signal { SIG_OFF = -1, SIG_RED = 0, SIG_YELLOW = 1, SIG_GREEN = 2 };
const Signal DEFAULT_SIGNAL = SIG_RED;

const char* const SIGNAL_TEXT[] = { "ALL OFF", "RED - STOP", "YELLOW - CAUTION", "GREEN - GO" };
// Console feedback, indexed by menu choice 0..5
const char* const CHOICE_TEXT[] = {
    "Exit", "Red / Stop", "Yellow / Caution", "Green / Go",
    "All Off", "Reset (back to Red / Stop)"
};

// program state
GLuint gVbo      = 0;
int    gWindow   = 0;
float  gAspect   = 500.0f / 700.0f;
Signal gActive   = DEFAULT_SIGNAL;   // touched by the GLUT thread only
int    gChanges  = 0;

Range gGround, gPole, gFrame, gBody;
Range gDisc[NUM_LIGHTS];
Range gHalo[NUM_LIGHTS];

mutex        gMutex;
queue<int>   gCommands;
atomic<bool> gQuit(false);

// Geometry builders
void addRect(vector<float>& v, float x0, float y0, float x1, float y1, Range& out)
{
    out.first = static_cast<GLint>(v.size() / 2);
    const float quad[12] = {
        x0, y0,  x1, y0,  x1, y1,     // triangle 1
        x0, y0,  x1, y1,  x0, y1      // triangle 2
    };
    v.insert(v.end(), quad, quad + 12);
    out.count = 6;
}

void addDisc(vector<float>& v, float cx, float cy, float radius, Range& out)
{
    out.first = static_cast<GLint>(v.size() / 2);
    for (int i = 0; i < DISC_SEGMENTS; ++i) {
        const float a0 = 2.0f * PI * i       / DISC_SEGMENTS;
        const float a1 = 2.0f * PI * (i + 1) / DISC_SEGMENTS;
        v.push_back(cx);                    v.push_back(cy);
        v.push_back(cx + radius * cos(a0)); v.push_back(cy + radius * sin(a0));
        v.push_back(cx + radius * cos(a1)); v.push_back(cy + radius * sin(a1));
    }
    out.count = DISC_SEGMENTS * 3;
}

// Build every vertex once and upload it into a single static VBO.
bool initBuffers()
{
    vector<float> v;

    const float housingH = 2.0f * HOUSING_PAD
                         + NUM_LIGHTS * 2.0f * LIGHT_RADIUS
                         + (NUM_LIGHTS - 1) * LIGHT_GAP;
    const float top = HOUSING_BOTTOM + housingH;

    addRect(v, -4.0f, -1.00f, 4.0f, -0.90f, gGround);                    // ground strip
    addRect(v, -0.04f, -0.90f, 0.04f, HOUSING_BOTTOM + 0.01f, gPole);    // pole
    addRect(v, -HOUSING_HALF_W - FRAME_BORDER, HOUSING_BOTTOM - FRAME_BORDER,
                HOUSING_HALF_W + FRAME_BORDER, top + FRAME_BORDER, gFrame); // metal frame
    addRect(v, -HOUSING_HALF_W, HOUSING_BOTTOM, HOUSING_HALF_W, top, gBody);// dark housing

    float centreY[NUM_LIGHTS];
    for (int i = 0; i < NUM_LIGHTS; ++i)
        centreY[i] = top - HOUSING_PAD - LIGHT_RADIUS - i * (2.0f * LIGHT_RADIUS + LIGHT_GAP);

    for (int i = 0; i < NUM_LIGHTS; ++i) addDisc(v, 0.0f, centreY[i], LIGHT_RADIUS, gDisc[i]);
    for (int i = 0; i < NUM_LIGHTS; ++i) addDisc(v, 0.0f, centreY[i], LIGHT_RADIUS * HALO_SCALE, gHalo[i]);

    glGenBuffers(1, &gVbo);
    glBindBuffer(GL_ARRAY_BUFFER, gVbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(v.size() * sizeof(float)),
                 v.data(),
                 GL_STATIC_DRAW);                 // geometry never changes; only colours do
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (glGetError() != GL_NO_ERROR) {
        cerr << "OpenGL error while creating the VBO.\n";
        return false;
    }
    return true;
}

void releaseBuffers()          // safe to call more than once
{
    if (gVbo != 0) {
        glDeleteBuffers(1, &gVbo);
        gVbo = 0;
    }
}

// rendering
inline void fill(const Range& r, float R, float G, float B, float A = 1.0f)
{
    glColor4f(R, G, B, A);
    glDrawArrays(GL_TRIANGLES, r.first, r.count);   // actual vertex count of the slice
}

void drawText(float x, float y, void* font, const string& s)
{
    glRasterPos2f(x, y);
    glutBitmapString(font, reinterpret_cast<const unsigned char*>(s.c_str()));
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindBuffer(GL_ARRAY_BUFFER, gVbo);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, static_cast<const void*>(0));

    fill(gGround, 0.20f, 0.22f, 0.25f);
    fill(gPole,   0.35f, 0.37f, 0.41f);
    fill(gFrame,  0.30f, 0.32f, 0.36f);
    fill(gBody,   0.04f, 0.04f, 0.05f);

    for (int i = 0; i < NUM_LIGHTS; ++i) {
        const bool on = (static_cast<int>(gActive) == i);
        const Color& c = LIGHT_ON[i];
        if (on) fill(gHalo[i], c.r, c.g, c.b, 0.28f);                // soft glow
        if (on) fill(gDisc[i], c.r, c.g, c.b);                       // lit
        else    fill(gDisc[i], c.r * DIM_FACTOR, c.g * DIM_FACTOR, c.b * DIM_FACTOR); // dark
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_BLEND);

    // On-screen status (text is not part of the required VBO geometry)
    glColor3f(0.92f, 0.92f, 0.92f);
    drawText(-gAspect + 0.05f, 0.945f, GLUT_BITMAP_HELVETICA_12,
             "Keys: 1 Red  2 Yellow  3 Green  4 Off  5 Reset  0 Exit");

    string status = string("Signal: ") + SIGNAL_TEXT[static_cast<int>(gActive) + 1]
                       + "   (changes: " + to_string(gChanges) + ")";
    drawText(-gAspect + 0.05f, -0.965f, GLUT_BITMAP_HELVETICA_18, status);

    glutSwapBuffers();
}

void reshape(int w, int h)
{
    if (h < 1) h = 1;
    gAspect = static_cast<float>(w) / static_cast<float>(h);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-gAspect, gAspect, -1.0, 1.0, -1.0, 1.0);   // keeps circles round
    glMatrixMode(GL_MODELVIEW);
}

// Menu logic
void refreshTitle()
{
    string t = string("Campus Traffic Signal Visualizer - ")
                  + SIGNAL_TEXT[static_cast<int>(gActive) + 1];
    glutSetWindowTitle(t.c_str());
}

void applyChoice(int choice)
{
    switch (choice) {
        case 1: gActive = SIG_RED;    ++gChanges; break;
        case 2: gActive = SIG_YELLOW; ++gChanges; break;
        case 3: gActive = SIG_GREEN;  ++gChanges; break;
        case 4: gActive = SIG_OFF;    ++gChanges; break;
        case 5: gActive = DEFAULT_SIGNAL; gChanges = 0; break;   // Reset
        case 0: gQuit = true; return;
        default: return;                                         // ignore anything else
    }
    refreshTitle();
}

void shutdownWindow()
{
    if (gWindow != 0) {
        const int w = gWindow;
        gWindow = 0;
        glutDestroyWindow(w);      // triggers onClose() while the context is valid
    }
    glutLeaveMainLoop();
}

void pollCommands(int)
{
    bool dirty = false;
    {
        lock_guard<mutex> lock(gMutex);
        while (!gCommands.empty()) {
            applyChoice(gCommands.front());
            gCommands.pop();
            dirty = true;
        }
    }
    if (gQuit) { shutdownWindow(); return; }
    if (dirty) glutPostRedisplay();
    glutTimerFunc(30, pollCommands, 0);
}

void keyboard(unsigned char key, int, int)
{
    if (key == 27) key = '0'; // ESC = Exit                      
    if (key >= '0' && key <= '5') {
        applyChoice(key - '0');
        if (gQuit) { shutdownWindow(); return; }
        glutPostRedisplay();
    }
}

void onClose()        // window closed by menu or by the X button
{
    gQuit = true;
    releaseBuffers(); // orderly shutdown: delete the VBO
}

// Console menu
void printMenu()
{
    cout << "\n==============================================\n"
                 "   CAMPUS TRAFFIC SIGNAL VISUALIZER - MENU\n"
                 "==============================================\n"
                 "  1  Red / Stop\n"
                 "  2  Yellow / Caution\n"
                 "  3  Green / Go\n"
                 "  4  All Off\n"
                 "  5  Reset (back to Red / Stop)\n"
                 "  0  Exit\n"
                 "----------------------------------------------\n";
}

// Accepts exactly one digit 0-5 (surrounding whitespace allowed).
bool parseChoice(const string& line, int& out)
{
    const size_t b = line.find_first_not_of(" \t\r\n");
    if (b == string::npos) return false;                 // empty input
    const size_t e = line.find_last_not_of(" \t\r\n");
    const string s = line.substr(b, e - b + 1);
    if (s.size() != 1 || s[0] < '0' || s[0] > '5') return false;
    out = s[0] - '0';
    return true;
}

void consoleLoop()
{
    printMenu();
    string line;
    while (!gQuit) {
        cout << "Enter choice (0-5): " << flush;
        if (!getline(cin, line)) break;             // stdin closed

        int choice = -1;
        if (!parseChoice(line, choice)) {
            cout << "  Invalid input. Please type a single number from 0 to 5.\n";
            continue;
        }
        {
            lock_guard<mutex> lock(gMutex);
            gCommands.push(choice);
        }
        cout << "  -> " << CHOICE_TEXT[choice] << "\n";
        if (choice == 0) break;
        printMenu();
    }
}

}   // namespace

int main(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 700);
    gWindow = glutCreateWindow("MP1 - Campus Traffic Signal Visualizer");

    const GLenum glewStatus = glewInit();                     // after the context exists
    if (glewStatus != GLEW_OK || !GLEW_VERSION_1_5) {
        cerr << "GLEW init failed (or OpenGL 1.5 / VBO support missing): "
                  << glewGetErrorString(glewStatus) << "\n";
        return 1;
    }

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glClearColor(0.07f, 0.09f, 0.13f, 1.0f);

    if (!initBuffers()) return 1;        // build + upload once

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutCloseFunc(onClose);
    glutTimerFunc(30, pollCommands, 0);
    refreshTitle();

    thread(consoleLoop).detach();        // console menu
    glutMainLoop();
    return 0;
}
