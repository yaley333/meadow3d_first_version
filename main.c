#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define PI 3.14159265358979323846f
#define ARRAY_GROW 2

typedef struct { float x,y,z; } Vec3;
typedef struct { float m[16]; } Mat4;
typedef struct { float px,py,pz,nx,ny,nz,r,g,b,flex; } Vertex;
typedef struct { GLuint vao,vbo; GLsizei count; } Mesh;
typedef struct { Vertex *data; size_t count,capacity; } VertexArray;

static GLFWwindow *g_window;
static int g_width=1280, g_height=800;
static int g_windowedX=100, g_windowedY=100, g_windowedWidth=1280, g_windowedHeight=800;
static int g_fullscreen=0;
static float g_camX=0.0f, g_camY=1.8f, g_camZ=8.0f;
static float g_yaw=-90.0f, g_pitch=-8.0f;
static double g_lastX=640.0, g_lastY=400.0;
static int g_firstMouse=1, g_mouseCaptured=1;
static float g_time=0.0f;
static GLuint g_program=0;
static GLuint g_skyProgram=0,g_skyVao=0;
static GLint u_model,u_view,u_projection,u_camera,u_time,u_wind,u_fogColor,u_fogNear,u_fogFar;
static GLint u_skyResolution,u_skyTime;
static Mesh g_terrain,g_grass,g_flowers;

static const char *VERTEX_SHADER =
"#version 330 core\n"
"layout(location=0) in vec3 aPos;\n"
"layout(location=1) in vec3 aNormal;\n"
"layout(location=2) in vec3 aColor;\n"
"layout(location=3) in float aFlex;\n"
"uniform mat4 uModel; uniform mat4 uView; uniform mat4 uProjection;\n"
"uniform vec3 uCamera; uniform float uTime; uniform float uWind;\n"
"out vec3 vColor; out vec3 vNormal; out vec3 vWorld;\n"
"void main(){\n"
" vec3 p=aPos;\n"
" // Subtle wind animation for tall vegetation. Ground vertices stay still.\n"
" float sway=sin(uTime*1.25 + p.x*0.8 + p.z*0.65)*0.11;\n"
" p.x += sway * uWind * aFlex;\n"
" vec4 world=uModel*vec4(p,1.0); vWorld=world.xyz;\n"
" vNormal=mat3(transpose(inverse(uModel)))*aNormal; vColor=aColor;\n"
" gl_Position=uProjection*uView*world; }\n";

static const char *FRAGMENT_SHADER =
"#version 330 core\n"
"in vec3 vColor; in vec3 vNormal; in vec3 vWorld;\n"
"uniform vec3 uCamera; uniform vec3 uFogColor; uniform float uFogNear; uniform float uFogFar;\n"
"out vec4 FragColor;\n"
"void main(){\n"
" vec3 N=normalize(vNormal); vec3 L=normalize(vec3(-0.45,0.82,0.32));\n"
" float diff=max(dot(N,L),0.0);\n"
" float hemi=0.5+0.5*clamp(N.y,0.0,1.0);\n"
" vec3 lit=vColor*(0.58+0.62*diff+0.16*hemi)+vec3(0.018,0.025,0.012);\n"
" float d=length(vWorld-uCamera); float fog=clamp((d-uFogNear)/(uFogFar-uFogNear),0.0,1.0); fog=fog*fog*(3.0-2.0*fog);\n"
" FragColor=vec4(mix(lit,uFogColor,fog),1.0); }\n";

static const char *SKY_VERTEX_SHADER =
"#version 330 core\n"
"out vec2 vUv;\n"
"void main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2)); vUv=p; gl_Position=vec4(p*2.0-1.0,1.0,1.0); }\n";

static const char *SKY_FRAGMENT_SHADER =
"#version 330 core\n"
"in vec2 vUv; uniform vec2 uResolution; uniform float uTime; out vec4 FragColor;\n"
"void main(){\n"
" vec2 uv=gl_FragCoord.xy/uResolution; float h=uv.y;\n"
" vec3 horizon=vec3(0.78,0.82,0.78); vec3 zenith=vec3(0.27,0.53,0.72);\n"
" vec3 sky=mix(horizon,zenith,smoothstep(0.38,0.96,h));\n"
" float sun=exp(-distance(uv,vec2(0.78,0.72))*18.0); float halo=exp(-distance(uv,vec2(0.78,0.72))*5.0);\n"
" sky+=vec3(0.25,0.19,0.10)*sun+vec3(0.10,0.10,0.07)*halo;\n"
" float cloud=sin(uv.x*15.0+sin(uv.y*9.0+uTime*0.012))*sin(uv.x*6.0-uv.y*13.0);\n"
" sky+=vec3(0.025,0.026,0.022)*smoothstep(0.18,0.72,cloud)*smoothstep(0.42,0.78,h);\n"
" FragColor=vec4(sky,1.0); }\n";

static float clampf(float x,float a,float b){ return x<a?a:(x>b?b:x); }
static Vec3 v3(float x,float y,float z){ Vec3 v={x,y,z}; return v; }
static Vec3 add3(Vec3 a,Vec3 b){return v3(a.x+b.x,a.y+b.y,a.z+b.z);}
static Vec3 sub3(Vec3 a,Vec3 b){return v3(a.x-b.x,a.y-b.y,a.z-b.z);}
static Vec3 mul3(Vec3 a,float s){return v3(a.x*s,a.y*s,a.z*s);}
static float dot3(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static Vec3 cross3(Vec3 a,Vec3 b){return v3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
static Vec3 norm3(Vec3 a){float l=sqrtf(dot3(a,a));return l>1e-6f?mul3(a,1.0f/l):v3(0,1,0);}

static Mat4 matIdentity(void){Mat4 r={{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}};return r;}
static Mat4 matMul(Mat4 a,Mat4 b){Mat4 r;int row,col,k;for(col=0;col<4;col++)for(row=0;row<4;row++){r.m[col*4+row]=0;for(k=0;k<4;k++)r.m[col*4+row]+=a.m[k*4+row]*b.m[col*4+k];}return r;}
static Mat4 matPerspective(float fov,float aspect,float nearP,float farP){Mat4 r={{0}};float f=1.0f/tanf(fov*0.5f);r.m[0]=f/aspect;r.m[5]=f;r.m[10]=(farP+nearP)/(nearP-farP);r.m[11]=-1;r.m[14]=(2*farP*nearP)/(nearP-farP);return r;}
static Mat4 matLookAt(Vec3 eye,Vec3 center,Vec3 up){Vec3 f=norm3(sub3(center,eye));Vec3 s=norm3(cross3(f,up));Vec3 u=cross3(s,f);Mat4 r=matIdentity();r.m[0]=s.x;r.m[4]=s.y;r.m[8]=s.z;r.m[1]=u.x;r.m[5]=u.y;r.m[9]=u.z;r.m[2]=-f.x;r.m[6]=-f.y;r.m[10]=-f.z;r.m[12]=-dot3(s,eye);r.m[13]=-dot3(u,eye);r.m[14]=dot3(f,eye);return r;}

static unsigned int hashU(unsigned int x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;return x;}
static float hash2(int x,int z){unsigned int n=hashU((unsigned int)x*374761393u ^ (unsigned int)z*668265263u);return (float)(n&0x00ffffffu)/16777215.0f;}
static float smoothNoise(float x,float z,float scale){float fx=x/scale,fz=z/scale;int ix=(int)floorf(fx),iz=(int)floorf(fz);float tx=fx-ix,tz=fz-iz;tx=tx*tx*(3-2*tx);tz=tz*tz*(3-2*tz);float a=hash2(ix,iz),b=hash2(ix+1,iz),c=hash2(ix,iz+1),d=hash2(ix+1,iz+1);float ab=a+(b-a)*tx,cd=c+(d-c)*tx;return ab+(cd-ab)*tz;}

static float terrainHeight(float x,float z){
    float h=0.16f*sinf(x*0.22f+z*0.11f)+0.12f*cosf(z*0.31f-x*0.15f);
    h+=0.20f*sinf(x*0.37f+z*0.16f)*cosf(z*0.29f-x*0.12f);
    h+=0.11f*sinf(x*0.61f-z*0.22f)*cosf(x*0.18f+z*0.44f);
    /* Near meadow rises gently into side hills, leaving a central valley. */
    h+=2.8f*expf(-((x+13)*(x+13))/72.0f-((z+8)*(z+8))/170.0f);
    h+=3.5f*expf(-((x-13)*(x-13))/90.0f-((z+10)*(z+10))/160.0f);
    h+=1.4f*expf(-(x*x)/24.0f-((z+20)*(z+20))/100.0f);
    h+=1.05f*expf(-((x+5.5f)*(x+5.5f))/34.0f-((z+13)*(z+13))/54.0f);
    h+=0.85f*expf(-((x-6.0f)*(x-6.0f))/42.0f-((z+18)*(z+18))/68.0f);
    /* Separate, irregular mountain peaks far behind the meadow. */
    h+=7.0f*expf(-((x+17)*(x+17))/42.0f-((z+43)*(z+43))/62.0f);
    h+=9.0f*expf(-((x+5)*(x+5))/35.0f-((z+47)*(z+47))/70.0f);
    h+=6.6f*expf(-((x-8)*(x-8))/48.0f-((z+44)*(z+44))/66.0f);
    h+=5.2f*expf(-((x-21)*(x-21))/70.0f-((z+40)*(z+40))/85.0f);
    float detail=(smoothNoise(x,z,4.2f)-0.5f)*0.8f+(smoothNoise(x,z,1.7f)-0.5f)*0.18f;
    float distant=clampf((-z-8.0f)/24.0f,0.0f,1.0f);
    h+=detail*(0.48f+0.85f*distant);
    return h;
}
static Vec3 terrainNormal(float x,float z){float e=0.12f;float dx=terrainHeight(x+e,z)-terrainHeight(x-e,z);float dz=terrainHeight(x,z+e)-terrainHeight(x,z-e);return norm3(v3(-dx,2*e,-dz));}

static void vaPush(VertexArray *a,Vertex v){if(a->count==a->capacity){size_t nc=a->capacity?a->capacity*ARRAY_GROW:4096;Vertex *p=(Vertex*)realloc(a->data,nc*sizeof(Vertex));if(!p){fprintf(stderr,"Out of memory building scene.\n");exit(2);}a->data=p;a->capacity=nc;}a->data[a->count++]=v;}
static Vertex vert(Vec3 p,Vec3 n,Vec3 c,float flex){Vertex v={p.x,p.y,p.z,n.x,n.y,n.z,c.x,c.y,c.z,flex};return v;}
static void triFlex(VertexArray *a,Vec3 p0,Vec3 p1,Vec3 p2,Vec3 color,float f0,float f1,float f2){Vec3 n=norm3(cross3(sub3(p1,p0),sub3(p2,p0)));vaPush(a,vert(p0,n,color,f0));vaPush(a,vert(p1,n,color,f1));vaPush(a,vert(p2,n,color,f2));}
static void tri(VertexArray *a,Vec3 p0,Vec3 p1,Vec3 p2,Vec3 color){triFlex(a,p0,p1,p2,color,0.0f,0.0f,0.0f);}
static void triNormalsFlex(VertexArray *a,Vec3 p0,Vec3 p1,Vec3 p2,Vec3 n0,Vec3 n1,Vec3 n2,Vec3 c0,Vec3 c1,Vec3 c2,float f0,float f1,float f2){vaPush(a,vert(p0,n0,c0,f0));vaPush(a,vert(p1,n1,c1,f1));vaPush(a,vert(p2,n2,c2,f2));}
static void triNormals(VertexArray *a,Vec3 p0,Vec3 p1,Vec3 p2,Vec3 n0,Vec3 n1,Vec3 n2,Vec3 c0,Vec3 c1,Vec3 c2){triNormalsFlex(a,p0,p1,p2,n0,n1,n2,c0,c1,c2,0.0f,0.0f,0.0f);}
static Mesh uploadMesh(VertexArray *a){Mesh m={0,0,0};glGenVertexArrays(1,&m.vao);glGenBuffers(1,&m.vbo);glBindVertexArray(m.vao);glBindBuffer(GL_ARRAY_BUFFER,m.vbo);glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)(a->count*sizeof(Vertex)),a->data,GL_STATIC_DRAW);glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)0);glEnableVertexAttribArray(1);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)(3*sizeof(float)));glEnableVertexAttribArray(2);glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)(6*sizeof(float)));glEnableVertexAttribArray(3);glVertexAttribPointer(3,1,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)(9*sizeof(float)));glBindVertexArray(0);m.count=(GLsizei)a->count;free(a->data);a->data=NULL;a->count=a->capacity=0;return m;}

static GLuint compileShader(GLenum type,const char *src){GLuint s=glCreateShader(type);glShaderSource(s,1,&src,NULL);glCompileShader(s);GLint ok=0;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);if(!ok){char log[4096];glGetShaderInfoLog(s,sizeof(log),NULL,log);fprintf(stderr,"Shader compile error:\n%s\n",log);glDeleteShader(s);return 0;}return s;}
static GLuint makeProgramFromSources(const char *vertexSource,const char *fragmentSource){GLuint vs=compileShader(GL_VERTEX_SHADER,vertexSource);GLuint fs=compileShader(GL_FRAGMENT_SHADER,fragmentSource);if(!vs||!fs)return 0;GLuint p=glCreateProgram();glAttachShader(p,vs);glAttachShader(p,fs);glLinkProgram(p);glDeleteShader(vs);glDeleteShader(fs);GLint ok=0;glGetProgramiv(p,GL_LINK_STATUS,&ok);if(!ok){char log[4096];glGetProgramInfoLog(p,sizeof(log),NULL,log);fprintf(stderr,"Program link error:\n%s\n",log);glDeleteProgram(p);return 0;}return p;}
static GLuint makeProgram(void){return makeProgramFromSources(VERTEX_SHADER,FRAGMENT_SHADER);}

static void buildTerrain(void){VertexArray a={0};const float start=-42.0f,end=42.0f,step=0.48f;int n=(int)((end-start)/step);for(int iz=0;iz<n;iz++){float z0=start+iz*step,z1=z0+step;for(int ix=0;ix<n;ix++){float x0=start+ix*step,x1=x0+step;Vec3 p[4]={v3(x0,terrainHeight(x0,z0),z0),v3(x1,terrainHeight(x1,z0),z0),v3(x1,terrainHeight(x1,z1),z1),v3(x0,terrainHeight(x0,z1),z1)};Vec3 normals[4]={terrainNormal(x0,z0),terrainNormal(x1,z0),terrainNormal(x1,z1),terrainNormal(x0,z1)};float noise=0.72f*smoothNoise(x0,z0,2.8f)+0.28f*smoothNoise(x0,z0,0.9f);float slope=1.0f-clampf((1.0f-norm3(normals[0]).y)*2.5f,0.0f,0.55f);Vec3 c=v3((0.075f+noise*0.035f)*slope,(0.235f+noise*0.105f)*slope,(0.035f+noise*0.022f)*slope);if(noise>0.90f)c=add3(c,v3(0.035f,0.045f,0.012f));triNormals(&a,p[0],p[2],p[1],normals[0],normals[2],normals[1],c,c,c);triNormals(&a,p[0],p[3],p[2],normals[0],normals[3],normals[2],c,c,c);}}g_terrain=uploadMesh(&a);}

static Vec3 bladePoint(Vec3 base,float t,float h,float leanX,float leanZ,float bendX,float bendZ){float bend=sinf(t*PI);return v3(base.x+leanX*t+bendX*bend,base.y+h*(t+0.035f*bend),base.z+leanZ*t+bendZ*bend);}

static void emitBlade(VertexArray *a,Vec3 base,float height,float leanX,float leanZ,float width,float bendX,float bendZ,Vec3 color,float flex){
    float dir=sqrtf(leanX*leanX+leanZ*leanZ);
    float sideX=dir>0.001f?-leanZ/dir:1.0f;
    float sideZ=dir>0.001f?leanX/dir:0.0f;
    float widths[4]={0.10f,0.78f,0.56f,0.015f};
    for(int segment=0;segment<3;segment++){
        float t0=(float)segment/3.0f,t1=(float)(segment+1)/3.0f;
        Vec3 c0=bladePoint(base,t0,height,leanX,leanZ,bendX,bendZ);
        Vec3 c1=bladePoint(base,t1,height,leanX,leanZ,bendX,bendZ);
        Vec3 l0=v3(c0.x-sideX*width*widths[segment],c0.y,c0.z-sideZ*width*widths[segment]);
        Vec3 r0=v3(c0.x+sideX*width*widths[segment],c0.y,c0.z+sideZ*width*widths[segment]);
        Vec3 l1=v3(c1.x-sideX*width*widths[segment+1],c1.y,c1.z-sideZ*width*widths[segment+1]);
        Vec3 r1=v3(c1.x+sideX*width*widths[segment+1],c1.y,c1.z+sideZ*width*widths[segment+1]);
        float shade=0.84f+0.16f*t1;
        Vec3 tint=mul3(color,shade);
        triFlex(a,l0,r0,l1,tint,flex*t0,flex*t0,flex*t1);
        triFlex(a,r0,r1,l1,tint,flex*t0,flex*t1,flex*t1);
    }
}

static void buildGrass(void){
    VertexArray a={0};
    const float spacing=0.34f;
    for(int iz=-250;iz<=58;iz++){
        float z=iz*spacing;
        float coverage=clampf((z+92.0f)/52.0f,0.20f,1.0f);
        int nx=(int)(39.0f*coverage/spacing);
        for(int ix=-nx;ix<=nx;ix++){
            if(hash2(ix+211,iz-93)<0.18f)continue;
            float x=ix*spacing+(hash2(ix+7,iz+19)-0.5f)*spacing*0.8f;
            float zz=z+(hash2(ix-11,iz+3)-0.5f)*spacing*0.8f;
            float y=terrainHeight(x,zz);
            float tuft=hash2(ix+81,iz-31);
            int blades=4+(int)(tuft*3.0f);
            for(int blade=0;blade<blades;blade++){
                float seed=hash2(ix*7+blade*19,iz*11-blade*23);
                float angle=seed*2.0f*PI;
                float spread=0.025f+0.075f*hash2(ix+blade*13,iz-blade*7);
                float height=(0.34f+0.70f*hash2(ix-blade*5,iz+blade*17))*coverage;
                float lean=0.10f+0.30f*hash2(ix+blade*29,iz-blade*3);
                float bend=(hash2(ix+blade*31,iz+blade*37)-0.5f)*0.12f;
                float hue=hash2(ix+blade*41,iz-blade*43);
                Vec3 green=hue<0.30f?v3(0.10f,0.38f,0.055f):hue<0.72f?v3(0.16f,0.53f,0.065f):v3(0.28f,0.60f,0.075f);
                Vec3 bladeBase=v3(x+cosf(angle)*spread,y,zz+sinf(angle)*spread);
                emitBlade(&a,bladeBase,height,cosf(angle)*lean,sinf(angle)*lean,0.035f+seed*0.025f,bend,bend*0.65f,green,0.8f+seed*0.2f);
            }
        }
    }
    g_grass=uploadMesh(&a);
}

static Vec3 flowerPlanePoint(Vec3 center,Vec3 axisU,Vec3 axisV,float u,float v){return add3(center,add3(mul3(axisU,u),mul3(axisV,v)));}

static void emitPetal(VertexArray *a,Vec3 center,Vec3 axisU,Vec3 axisV,float angle,float size,Vec3 color,float flex){
    float c=cosf(angle),s=sinf(angle);
    Vec3 radial=add3(mul3(axisU,c),mul3(axisV,s));
    Vec3 across=add3(mul3(axisU,-s),mul3(axisV,c));
    Vec3 root=flowerPlanePoint(center,radial,across,size*0.06f,0.0f);
    Vec3 left=flowerPlanePoint(center,radial,across,size*0.57f,-size*0.24f);
    Vec3 tip=flowerPlanePoint(center,radial,across,size,0.0f);
    Vec3 right=flowerPlanePoint(center,radial,across,size*0.57f,size*0.24f);
    Vec3 light=v3(fminf(color.x+0.08f,1.0f),fminf(color.y+0.08f,1.0f),fminf(color.z+0.08f,1.0f));
    triFlex(a,root,left,tip,color,flex,flex,flex*0.9f);
    triFlex(a,root,tip,right,light,flex,flex*0.9f,flex);
}

static void buildFlowers(void){
    VertexArray a={0};
    const Vec3 axisU=v3(1.0f,0.0f,0.0f),axisV=v3(0.0f,0.72f,-0.69f);
    for(int iz=-230;iz<=48;iz++){
        float z=iz*0.40f;
        float coverage=clampf((z+78.0f)/48.0f,0.20f,1.0f);
        int nx=(int)(34.0f*coverage/0.40f);
        for(int ix=-nx;ix<=nx;ix++){
            float pick=hash2(ix+601,iz-403);
            float x=ix*0.40f+(hash2(ix+9,iz+13)-0.5f)*0.32f;
            float zz=z+(hash2(ix-7,iz+5)-0.5f)*0.32f;
            float patch=smoothNoise(x,zz,3.4f);
            if((patch<0.34f&&pick<0.94f)||(patch>=0.34f&&pick<0.57f))continue;
            float y=terrainHeight(x,zz);
            float size=(0.12f+0.12f*hash2(ix+17,iz-21))*coverage;
            int type=(int)(hash2(ix-22,iz+8)*6.0f);
            Vec3 petal=type==0?v3(1.0f,0.20f,0.10f):type==1?v3(1.0f,0.72f,0.08f):type==2?v3(0.95f,0.20f,0.48f):type==3?v3(0.94f,0.88f,0.70f):type==4?v3(0.72f,0.34f,0.92f):v3(0.28f,0.66f,0.98f);
            int petals=type==0?6:type==1?8:type==2?7:type==3?10:type==4?5:9;
            float stemH=(0.30f+0.28f*hash2(ix+37,iz-14))*coverage;
            Vec3 base=v3(x,y,zz);
            Vec3 head=v3(x+0.025f*(hash2(ix+3,iz)-0.5f),y+stemH,zz);
            Vec3 stemGreen=v3(0.07f,0.32f,0.035f);
            triFlex(&a,base,v3(x+0.018f,y+stemH*0.72f,zz),v3(x-0.018f,y+stemH*0.72f,zz),stemGreen,0.0f,0.55f,0.55f);
            float leafSide=(hash2(ix+91,iz+67)-0.5f)*0.08f;
            Vec3 leafTip=v3(x+leafSide,y+stemH*0.56f,zz-0.10f);
            triFlex(&a,v3(x,y+stemH*0.30f,zz),leafTip,v3(x+0.035f,y+stemH*0.42f,zz),v3(0.10f,0.42f,0.035f),0.2f,0.6f,0.4f);
            for(int k=0;k<petals;k++){
                float angle=2.0f*PI*(float)k/(float)petals+0.08f*hash2(ix+k,iz-k);
                emitPetal(&a,head,axisU,axisV,angle,size*(0.76f+0.35f*hash2(ix-k*3,iz+k*5)),petal,0.75f);
            }
            Vec3 centerColor=type==1?v3(0.56f,0.28f,0.035f):v3(1.0f,0.76f,0.12f);
            float centerSize=size*0.14f;
            triFlex(&a,flowerPlanePoint(head,axisU,axisV,-centerSize,0),flowerPlanePoint(head,axisU,axisV,centerSize,0),flowerPlanePoint(head,axisU,axisV,0,centerSize),centerColor,0.75f,0.75f,0.75f);
            triFlex(&a,flowerPlanePoint(head,axisU,axisV,centerSize,0),flowerPlanePoint(head,axisU,axisV,-centerSize,0),flowerPlanePoint(head,axisU,axisV,0,-centerSize),centerColor,0.75f,0.75f,0.75f);
        }
    }
    g_flowers=uploadMesh(&a);
}

static void mouseCallback(GLFWwindow *window,double xpos,double ypos){(void)window;if(g_firstMouse){g_lastX=xpos;g_lastY=ypos;g_firstMouse=0;}float dx=(float)(xpos-g_lastX),dy=(float)(g_lastY-ypos);g_lastX=xpos;g_lastY=ypos;if(g_mouseCaptured){g_yaw+=dx*0.10f;g_pitch+=dy*0.10f;g_pitch=clampf(g_pitch,-55.0f,35.0f);}}
static void toggleFullscreen(GLFWwindow *window){if(g_fullscreen){glfwSetWindowMonitor(window,NULL,g_windowedX,g_windowedY,g_windowedWidth,g_windowedHeight,GLFW_DONT_CARE);g_fullscreen=0;return;}glfwGetWindowPos(window,&g_windowedX,&g_windowedY);glfwGetWindowSize(window,&g_windowedWidth,&g_windowedHeight);GLFWmonitor *monitor=glfwGetPrimaryMonitor();if(!monitor)return;const GLFWvidmode *mode=glfwGetVideoMode(monitor);if(!mode)return;glfwSetWindowMonitor(window,monitor,0,0,mode->width,mode->height,mode->refreshRate);g_fullscreen=1;}
static void keyCallback(GLFWwindow *window,int key,int scancode,int action,int mods){(void)scancode;(void)mods;if(key==GLFW_KEY_ESCAPE&&action==GLFW_PRESS){if(g_mouseCaptured){g_mouseCaptured=0;glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_NORMAL);}else glfwSetWindowShouldClose(window,1);}if(key==GLFW_KEY_TAB&&action==GLFW_PRESS){g_mouseCaptured=!g_mouseCaptured;glfwSetInputMode(window,GLFW_CURSOR,g_mouseCaptured?GLFW_CURSOR_DISABLED:GLFW_CURSOR_NORMAL);g_firstMouse=1;}if(key==GLFW_KEY_F&&action==GLFW_PRESS)toggleFullscreen(window);}
static void framebufferCallback(GLFWwindow *window,int w,int h){(void)window;if(h<1)h=1;g_width=w;g_height=h;glViewport(0,0,w,h);}

static void updateCamera(float dt){Vec3 forward=v3(cosf(g_yaw*PI/180.0f)*cosf(g_pitch*PI/180.0f),sinf(g_pitch*PI/180.0f),sinf(g_yaw*PI/180.0f)*cosf(g_pitch*PI/180.0f));forward.y=0;forward=norm3(forward);Vec3 right=norm3(cross3(forward,v3(0,1,0)));float speed=(glfwGetKey(g_window,GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS?9.0f:4.2f)*dt;if(glfwGetKey(g_window,GLFW_KEY_W)==GLFW_PRESS){g_camX+=forward.x*speed;g_camZ+=forward.z*speed;}if(glfwGetKey(g_window,GLFW_KEY_S)==GLFW_PRESS){g_camX-=forward.x*speed;g_camZ-=forward.z*speed;}if(glfwGetKey(g_window,GLFW_KEY_A)==GLFW_PRESS){g_camX-=right.x*speed;g_camZ-=right.z*speed;}if(glfwGetKey(g_window,GLFW_KEY_D)==GLFW_PRESS){g_camX+=right.x*speed;g_camZ+=right.z*speed;}g_camX=clampf(g_camX,-38,38);g_camZ=clampf(g_camZ,-38,25);g_camY=terrainHeight(g_camX,g_camZ)+1.65f;if(glfwGetKey(g_window,GLFW_KEY_SPACE)==GLFW_PRESS)g_camY+=1.0f;}

static void drawMesh(Mesh m){glBindVertexArray(m.vao);glDrawArrays(GL_TRIANGLES,0,m.count);}

int main(void){
    if(!glfwInit()){fprintf(stderr,"GLFW initialization failed.\n");return 1;}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GLFW_TRUE);glfwWindowHint(GLFW_SAMPLES,4);
    g_window=glfwCreateWindow(g_width,g_height,"Wildflower Valley - OpenGL 3.3",NULL,NULL);if(!g_window){fprintf(stderr,"Could not create OpenGL 3.3 window. Update your graphics driver.\n");glfwTerminate();return 1;}
    glfwMakeContextCurrent(g_window);glfwSwapInterval(1);
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){fprintf(stderr,"GLAD could not load OpenGL functions.\n");glfwDestroyWindow(g_window);glfwTerminate();return 1;}
    printf("OpenGL: %s\n",glGetString(GL_VERSION));
    glfwSetCursorPosCallback(g_window,mouseCallback);glfwSetKeyCallback(g_window,keyCallback);glfwSetFramebufferSizeCallback(g_window,framebufferCallback);glfwSetInputMode(g_window,GLFW_CURSOR,GLFW_CURSOR_DISABLED);
    glEnable(GL_DEPTH_TEST);glDepthFunc(GL_LEQUAL);glEnable(GL_MULTISAMPLE);glDisable(GL_CULL_FACE);
    g_program=makeProgram();g_skyProgram=makeProgramFromSources(SKY_VERTEX_SHADER,SKY_FRAGMENT_SHADER);if(!g_program||!g_skyProgram){glfwDestroyWindow(g_window);glfwTerminate();return 1;}
    u_model=glGetUniformLocation(g_program,"uModel");u_view=glGetUniformLocation(g_program,"uView");u_projection=glGetUniformLocation(g_program,"uProjection");u_camera=glGetUniformLocation(g_program,"uCamera");u_time=glGetUniformLocation(g_program,"uTime");u_wind=glGetUniformLocation(g_program,"uWind");u_fogColor=glGetUniformLocation(g_program,"uFogColor");u_fogNear=glGetUniformLocation(g_program,"uFogNear");u_fogFar=glGetUniformLocation(g_program,"uFogFar");
    u_skyResolution=glGetUniformLocation(g_skyProgram,"uResolution");u_skyTime=glGetUniformLocation(g_skyProgram,"uTime");glGenVertexArrays(1,&g_skyVao);
    fprintf(stderr,"Generating terrain and vegetation meshes...\n");buildTerrain();buildGrass();buildFlowers();fprintf(stderr,"Ready. WASD move, mouse look, Shift run, Tab release/capture mouse, Esc exit.\n");
    Mat4 model=matIdentity();double last=glfwGetTime();
    while(!glfwWindowShouldClose(g_window)){double now=glfwGetTime();float dt=(float)(now-last);last=now;if(dt>0.05f)dt=0.05f;g_time=(float)now;glfwPollEvents();updateCamera(dt);
        float yr=g_yaw*PI/180.0f,pr=g_pitch*PI/180.0f;Vec3 front=v3(cosf(yr)*cosf(pr),sinf(pr),sinf(yr)*cosf(pr));Vec3 eye=v3(g_camX,g_camY,g_camZ);Mat4 view=matLookAt(eye,add3(eye,front),v3(0,1,0));Mat4 proj=matPerspective(65.0f*PI/180.0f,(float)g_width/(float)(g_height?g_height:1),0.1f,180.0f);
        glClearColor(0.72f,0.78f,0.77f,1.0f);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);glUseProgram(g_skyProgram);glUniform2f(u_skyResolution,(float)g_width,(float)g_height);glUniform1f(u_skyTime,g_time);glBindVertexArray(g_skyVao);glDrawArrays(GL_TRIANGLES,0,3);glUseProgram(g_program);glUniformMatrix4fv(u_model,1,GL_FALSE,model.m);glUniformMatrix4fv(u_view,1,GL_FALSE,view.m);glUniformMatrix4fv(u_projection,1,GL_FALSE,proj.m);glUniform3f(u_camera,eye.x,eye.y,eye.z);glUniform1f(u_time,g_time);glUniform3f(u_fogColor,0.70f,0.76f,0.72f);glUniform1f(u_fogNear,20.0f);glUniform1f(u_fogFar,92.0f);
        glUniform1f(u_wind,0.0f);drawMesh(g_terrain);glUniform1f(u_wind,1.0f);drawMesh(g_grass);drawMesh(g_flowers);glBindVertexArray(0);glUseProgram(0);glfwSwapBuffers(g_window);
    }
    glDeleteVertexArrays(1,&g_skyVao);glDeleteVertexArrays(1,&g_terrain.vao);glDeleteBuffers(1,&g_terrain.vbo);glDeleteVertexArrays(1,&g_grass.vao);glDeleteBuffers(1,&g_grass.vbo);glDeleteVertexArrays(1,&g_flowers.vao);glDeleteBuffers(1,&g_flowers.vbo);glDeleteProgram(g_skyProgram);glDeleteProgram(g_program);glfwDestroyWindow(g_window);glfwTerminate();return 0;
}
