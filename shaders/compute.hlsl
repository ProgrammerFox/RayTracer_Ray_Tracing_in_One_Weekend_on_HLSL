float fract(float x)
{
	return x - floor(x);
}

float2 fract(float2 x)
{
	return float2(x.x - floor(x.x), x.y - floor(x.y));
}

float3 fract(float3 x)
{
	return float3(x.x - floor(x.x), x.y - floor(x.y), x.z - floor(x.z));
}

float4 fract(float4 x)
{
	return float4(x.x - floor(x.x), x.y - floor(x.y), x.z - floor(x.z), x.w - floor(x.w));
}

cbuffer varBuffer : register(b0) {
	float roll;
	float4 CamPos;
	float2 ScrSize;
	float4 LastCamPos;
	int reBound;
}

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
	uint b = (((z << S1) ^ z) >> S2);
	return (((z & M) << S3) ^ b);	
}

uint LCGStep(uint z, uint A, uint C)
{
	return (A * z + C);	
}

float2 hash22(float2 p, float2 u_seed1)
{
	p += u_seed1.x;
	float3 p3 = fract(p.xyx * float3(0.1031f, 0.1030f, 0.0973f));
	p3 += dot(p3, p3.yzx + 33.33f);
	return fract((p3.xx + p3.yz) * p3.zy);
}

struct Random
{
	float2 u_seed1;
	float2 u_seed2;
	uint4 R_STATE;
	float nextfloat()
	{
		//return 0.7;
		R_STATE.x = TausStep(R_STATE.x, 13, 19, 12, uint(4294967294));
		R_STATE.y = TausStep(R_STATE.y, 2, 25, 4, uint(4294967288));
		R_STATE.z = TausStep(R_STATE.z, 3, 11, 17, uint(4294967280));
		R_STATE.w = LCGStep(R_STATE.w, uint(1664525), uint(1013904223));
		//return 0.6;
		
		return fract(roll + 2.3283064365387e-10f * float((R_STATE.x ^ R_STATE.y ^ R_STATE.z ^ R_STATE.w)));
	}
};

float rand_1_04( float2 uv )
{
    return 1 - frac(sin(dot(uv, float2(12.9898, 78.233))) * 43758.5453);
}

float rand_1_05(in float2 uv)
{
    float2 noise = (frac(sin(dot(uv ,float2(12.9898,78.233)*2.0)) * 43758.5453));
    return abs(noise.x + noise.y) * 0.5;
}



struct Sphere
{
	float4 Geometry;
	int4 MaterialIndex;
};

struct Box
{
	float4 Position;
	float4 Scale;
	int4 MaterialIndex;
};

struct HitRecord
{
	float t;
	float3 pos;
	uint type;
	uint MatIdx;
	float3 Normal;
	bool FrontFace;
	void SetFaceNormal(float3 rd, float3 n)
	{
		FrontFace = dot(rd, n) < 0;
		Normal = FrontFace ? n : -n;
	}
};

bool SphereHit(float3 ro, float3 rd, float4 sphere,  out HitRecord rec)
{
	
	//if(radius < 0.5f) return false;
	float3 oc = ro - sphere.xyz;
	float b = dot(oc, rd);
	float c = dot(oc, oc) - sphere.w * sphere.w;
	float d = b * b - c;
	//if(dot(oc, rd) - dot(oc, oc) > -2.01f) return false;
	if(d < 0.0f) return false;
	d = sqrt(d);
	rec.t = -b - d;
	rec.type = 1;
	rec.pos = ro + rd * rec.t;
	float3 norm = (rec.pos - sphere.xyz) / sphere.w; 
	rec.SetFaceNormal(rd, norm);
	return true;
}
bool BoxHit( float3 ro1, float3 rd, float3 boxPos, float3 boxSize, out HitRecord rec) 
{
	float3 ro = ro1 - boxPos;
    float3 m = 1.0 / rd; // can precompute if traversing a set of aligned boxes
    float3 n = m * ro;   // can precompute if traversing a set of aligned boxes
    float3 k = abs(m) * boxSize;
    float3 t1 = -n - k;
    float3 t2 = -n + k;
    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN > tF || tF < 0.0) return false; // no intersection
    
    float3 norm = -sign(rd)*step(t1.yzx,t1.xyz) * step(t1.zxy,t1.xyz);
    //float3 norm = 1;
    rec.SetFaceNormal(rd, norm);
    rec.t = tN;
    rec.type = 1;
    rec.pos = ro1 + rd * rec.t;
    return true;
}

float3 RandomInSphere(Random random)
{
	float3 rand = float3(random.nextfloat(), random.nextfloat(), random.nextfloat());
	float theta = rand.x * 2.0 * 3.14159265;
	float v = rand.y;
	float phi = acos(2.0 * v - 1.0);
	float r = pow(rand.z, 1.0 / 3.0);
	float x = r * sin(phi) * cos(theta);
	float y = r * sin(phi) * sin(theta);
	float z = r * cos(phi);
	return float3(x, y, z);
}

float3 RandomInHemisphere(Random random, float3 Normal)
{
	float3 rand = float3(random.nextfloat(), random.nextfloat(), random.nextfloat());
	float theta = rand.x * 2.0 * 3.14159265;
	float v = rand.y;
	float phi = acos(2.0 * v - 1.0);
	float r = pow(rand.z, 1.0 / 3.0);
	float x = r * sin(phi) * cos(theta);
	float y = r * sin(phi) * sin(theta);
	float z = r * cos(phi);
	float3 res = float3(x, y, z);
	if(dot(res, Normal) < 0) res = - res;
	return res;
}

float3 RandomInDisk(Random random)
{
	float r = sqrt(random.nextfloat());
	float theta = random.nextfloat() * 2 * 3.14159265f;
	float x = r * cos(theta);
	float y = r * sin(theta);
	return float3(x, y, 0);
}

struct Camera
{
	float3 viewport;
	float3 aspect_ratio;
	float3 origin;
	float3 horizontal;
	float3 vertical;
	float3 lower_left_corner;
	float3 lens_radius;
	float3 w;
	float3 u;
	float3 v;
	void SetCamera(float2 uv, float3 look_from, float focus_dist, float aperture)
	{
		
		origin = look_from;
		float3 look_to = 0;
		float3 vup = float3(0, 1, 0);
		
		//float aspect_ratio = uv.x / uv.y;
		float h = 1;//tan(1.57f / 2);
		float viewport_height = 2.0 * h;
        float viewport_width = viewport_height;
	
		w = normalize(look_from - look_to);
		u = normalize(cross(vup, w));
		v = cross(w, u);
	

		vertical = 0.82 * viewport_height * focus_dist * v;
		
		horizontal = 0.82 * viewport_width * focus_dist * u;
    	lower_left_corner = look_from  - vertical/2 -  horizontal/2 - focus_dist * w;
    	
    	lens_radius = aperture / 2;
    	
	}
	float3 GetRdRay(float2 uv, Random rand, out float3 ro)
	{
		float2 rnd = float2((rand.nextfloat() - 0.5) / 500, (rand.nextfloat() - 0.5) / 500);
		float2 rd = lens_radius * RandomInDisk(rand).xy + rnd;
		float3 offset = u * rd.x + v * rd.y;
		
		ro = origin + offset;
		
		return normalize(lower_left_corner + (uv.x + rnd.x) * horizontal + (uv.y + rnd.y + 0.22f) * vertical - origin - offset);
	}
	
};

float reflectance(float cosine, float ref_idx) 
{

    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

struct Material
{
	float4 LayersValue;
	float4 Color;
	
	
	float3 Scatter(float3 ro, float3 rd, HitRecord rec, Random rand, out int FaceInfo)
	{
		float3 norm = rec.Normal * (rec.FrontFace ? 1 : -1);
		float3 reflected = reflect(rd, norm);
		float3 res1 = LayersValue.x <= rand.nextfloat() ? (norm + RandomInSphere(rand)) : RandomInHemisphere(rand, norm);
		float3 res2 = reflected + RandomInHemisphere(rand, norm) * LayersValue.y;
		float3 res3 = 0;
		
		FaceInfo = 1;
		if(LayersValue.w > 0)
		{
			FaceInfo = -1;
			float ir = LayersValue.w;
			float refraction_ratio = rec.FrontFace ? (1.0 / ir) : ir;

			//float aaa = 1.0 / ir;

            //float3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = min(dot(-rd, rec.Normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            float3 direction = 0;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rand.nextfloat())
                direction = reflect(rd, rec.Normal);
            else
                direction = refract(rd, rec.Normal, refraction_ratio);

            //ro = rec.pos;
            res3 = direction;
		}
		
		if(LayersValue.w <= 0) return ((LayersValue.z <= rand.nextfloat()) ? res2 : res1);
		else return res3;
	}
};

RWTexture2D<float4> img_output : register(u0);

RWStructuredBuffer<Sphere> Spheres : register(u1);

RWStructuredBuffer<Material> Materials : register(u2);

RWTexture2D<float4> output_image : register(u3);

RWStructuredBuffer<Box> Boxes : register(u4);

bool WorldHit(float3 ro, float3 rd, uint SpheresCount, uint BoxesCount, out HitRecord rec)
{
	HitRecord tmp, res;
	
	bool WasHit = false;
	float MaxDist = 3.402823466e+38 - 1, MinDist = 0.001f;
	
	bool ll = false;
	
	uint idxmat = 0;
	
	for(int i = 0; i < SpheresCount; i++)
	{
		if(SphereHit(ro, rd, Spheres[i].Geometry, tmp)) 
		{
			//tmp.MatIdx = 1;
			
			if(tmp.t < MaxDist && tmp.t > MinDist)
			{
				WasHit = true;
				res = tmp;
				MaxDist = tmp.t;
				ll = true;
				idxmat = i;
				
			}
			
		}
	}
	if(ll) res.MatIdx = Spheres[idxmat].MaterialIndex.x;
	ll = false;
	for(int i = 0; i < BoxesCount; i++)
	{
		if(BoxHit(ro, rd, Boxes[i].Position, Boxes[i].Scale, tmp)) 
		{
			//tmp.MatIdx = 1;
			
			if(tmp.t < MaxDist && tmp.t > MinDist)
			{
				WasHit = true;
				res = tmp;
				MaxDist = tmp.t;
				ll = true;
				idxmat = i;
				
			}
			
		}
	}
	
	if(ll) res.MatIdx = Boxes[idxmat].MaterialIndex.x;
	
	rec = res;
	return WasHit;
}

float3 ray_color(float3 ro, float3 rd, float2 uv, Random rand)
{
	uint SpheresCount, BoxesCount, tmp1;
	Spheres.GetDimensions(SpheresCount, tmp1);
	Boxes.GetDimensions(BoxesCount, tmp1);
	//HitRecord rec;
	
	float3 resColor = 1;
	
	float lostray = 1;
	//bool raywaslost = false;
	
	int rBound = reBound;
	if(rBound < 1) rBound = 1;
	else if(rBound > 100) rBound = 100;
	
	int lastindex = 0;
	
	for(int i = 0; i < rBound; i++)
	{
		HitRecord rec;
		
		
		if(WorldHit(ro, rd, SpheresCount, BoxesCount, rec))  
		{
			float FaceInfo;
			
			rd = normalize(Materials[rec.MatIdx].Scatter(ro, rd, rec, rand, FaceInfo));
			ro = rec.pos + rec.Normal * (0.0001f) * (rec.FrontFace ? 1 : -1) * FaceInfo;
			float4 mat_col = Materials[rec.MatIdx].Color;
			resColor = resColor * mat_col.xyz;
			if(mat_col.w >= 1.0f) return resColor;
			//return Materials[Spheres[res.MatIdx].MaterialIndex.x].Color;
		}
		else break;

		
		
		//if(i >= rBound - 1) lostray = 0;
	}
	HitRecord rec;
	//if(WorldHit(ro, rd, SpheresCount, BoxesCount, rec)) return 0;
	
	//if(lastindex >= rBound - 1) lostray = 0;
	
	//for(int i = 0; i < SpheresCount; i++)
	//{
	//	if(SphereHit(ro, rd, Spheres[i].Geometry, rec)) return 0;
	//}
	
	return 0;
	//return resColor * 0;
	float t = 0.5f * (rd.y + 1);
	return (resColor * ((1 - t) * float3(1) + t * float3(0.2f, 0.4f, 1))) * lostray;
}



[numthreads(16, 16, 1)]
void main( uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID )
{
    uint2 ScreenSize;
	img_output.GetDimensions(ScreenSize.x, ScreenSize.y);

    float2 uv = dispatchThreadID.xy / (float)ScreenSize;
    float2 cam_uv = uv;//float2((uv.x - 0.5) * (float)ScrSize.x / (float)ScrSize.y + 0.5, uv.y);
	
	uv.y += 5;
	uv.x += 5;
	uv *= 5;
	
	//uv = 1 / roll;
	
	float rd1 = rand_1_05(float2(1.0 / (1 - uv.x) * roll, 1.0 / (1 - uv.y) * roll));
	float rd2 = rand_1_04(float2(rd1 / (1 - uv.y) - roll, rd1 / (1 - uv.x) - roll));
	float rd3 = rand_1_05(float2(rd2 / (1 -  rd1) - roll, rd1 / (1 -  rd2) - roll));
	
	//float rd1 = roll, rd3 = roll;
							
	Random rand;
	rand.u_seed1 = float2(1/rd1, 1/rd3 * 999);
	rand.u_seed2 = float2(1/rd3, 1/rd1 * 999);
	float2 uvRes = hash22(uv + 1.0, rand.u_seed1) * ScreenSize + ScreenSize;
	rand.R_STATE = uint4(rand.u_seed1.x + uvRes.x, rand.u_seed1.y + uvRes.x, rand.u_seed2.x + uvRes.y, rand.u_seed2.y + uvRes.y);
    rand.nextfloat();
    
    uint SpheresCount, BoxesCount, tmp1;
	Spheres.GetDimensions(SpheresCount, tmp1);
	Boxes.GetDimensions(BoxesCount, tmp1);
	
    float3 FocusRo = CamPos.xyz, FocusRd = normalize(float3(0, -0.7, 0)-CamPos.xyz);
    HitRecord FocusRec;
    
    WorldHit(FocusRo, FocusRd, SpheresCount, BoxesCount, FocusRec);
    float FocusDst = length(CamPos.xyz - FocusRec.pos) + 0.5;
    
    Camera mainCamera;
    mainCamera.SetCamera(float2(ScreenSize), CamPos.xyz, FocusDst, 0.01);//CamPos.xyz
    
    if((LastCamPos - CamPos).x == 0 && (LastCamPos - CamPos).y == 0 && (LastCamPos - CamPos).z == 0);
    else img_output[dispatchThreadID.xy] = float4(0);
    
    float3 ro; 
    float3 rd = mainCamera.GetRdRay(cam_uv, rand, ro);
    
    float3 Color = ray_color(ro, rd, uv, rand);
    
    img_output[dispatchThreadID.xy] += float4(Color, 1);
    output_image[dispatchThreadID.xy] = sqrt(img_output[dispatchThreadID.xy] / img_output[dispatchThreadID.xy].w);
}
