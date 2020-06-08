---
title: Path Tracing Fog
date: 2018-05-28 10:00:00 -07:00
tags: [c++, cpu, gpu, ray-tracing, path-tracing, volumetric-scattering]
description: Utilizes volumetric scattering and ray marching
image: /assets/img/path-tracing-fog/1.jpg
---

## Rendering Fog

### Introduction

The goal of this project was to capture realistic shadows in between the shadow on the floor and the object (Stanford dragon).

The following image depicts the type of effect desired for this project (volumetric scattering):
<figure>
<img src="/assets/img/path-tracing-fog/1.jpg" alt="attic">
<figcaption>stock photo of dust particles</figcaption>
</figure>
Although the picture shows dust, generating fog is very similar.

### Images

Final image:
<figure>
<img src="/assets/img/path-tracing-fog/final.png" alt="final">
<figcaption>Final path traced image</figcaption>
</figure>

Without the effect:
<figure>
<img src="/assets/img/path-tracing-fog/original.png" alt="original">
<figcaption>Control example</figcaption>
</figure>

Other generated examples:
<figure>
<img src="/assets/img/path-tracing-fog/good.png" alt="good">
</figure>
<figure>
<img src="/assets/img/path-tracing-fog/nice.png" alt="nice">
</figure>

### How does it work?

I used volumetric scattering to get this effect.

<figure>
<img src="/assets/img/path-tracing-fog/eq.png" alt="equation">
<figcaption>equation used</figcaption>
</figure>

This equation gives the light at location (x) plus some direction (omega) times some distance (s). L(x,w) is the light at the previous location, the exponential coefficient is the amount of extinction which tells us how much light we lose to absorption and out scattering, E(x) is the emission of the volume at location x, and finally L sub i of x is the in scattering at x.

```c++
void CalculateColor(
        Ray & ray,
        Color & col,
        Color & inScattering,
        float distance)
{
    glm::vec3 x = ray.Origin;

    const float extinctionCoeff = 0.9f;
    distance = distance / stepSize;

    const float e = 2.71828f;
    float extinction = powf(e, -extinctionCoeff*distance);
    Color emission = GetEmission(x);
    col.AddScaled(emission, distance);
    col.AddScaled(inScattering, distance);
    col.Scale(extinction);
}
```

To use this equation, you must ray march through the volume. Start at the first intersection at the volume and in small random increments we use the light at the previous location (L(x,w)) to get the light at the next position (L(x + s*w, w)) each increment. When we exit the volume we have an accumulated light value going through the volume and we use that light with the light we ray traced. I personally did this by setting the initial light to white going through the volume, and then I do the normal ray tracing calculation for secondary rays and shadow rays, then finally multiply this color with the final color that came out of the volume.

This is easy up until the calculation for in scattering. In scattering was calculated in this project by first shooting rays to each light source and if there was a path the color of the light scaled by intensity times the probability of the ray going that direction was added to the color of in scattering. To determine the probability, the ray went towards a given light source, I got the cosine between the normalized rays (just the dot product) and plugged it into the Henyey-Greenstein Function with a g value of 0.3 where g determines anisotropy. Next I shoot N number (2 or 3) of rays in random directions and get colors of either the sky or objects these rays hit, then of course multiply it by the probability the ray can go in that direction and then divide it by N.

Code:
```c++
Color marchCol = Color::WHITE;

float tmin, tmax;
Ray marchRay = ray;
if (VolumeIntersection(marchRay, tmin, tmax))
{
    glm::vec3 start = tmin * marchRay.Direction + marchRay.Origin;
    glm::vec3 end = tmax * marchRay.Direction + marchRay.Origin;
    marchRay.Origin = start;
    float thedot = glm::dot(marchRay.Origin - start,
        marchRay.Origin - end);
    while (thedot < 0.00001f)
    {
        Ray newRay = StepRay(marchRay);

        if (glm::dot(marchRay.Direction, hitpos - newRay.Origin) <= 0)
        {
            break;
        }

        float distToNewRay = glm::distance(marchRay.Origin, newRay.Origin);

        Color lightComingIn = Color::BLACK;

        Ray rayToLight;
        rayToLight.Origin = newRay.Origin;
        int numLights = Scn->GetNumLights();
        for (int i = 0; i < numLights; i++)
        {
            Color col;
            glm::vec3 toLight, ltPos;
            float intensity = Scn->GetLight(i).Illuminate(newRay.Origin,
                col, toLight, ltPos);

            if (!intensity)
            {
                continue;
            }

            Intersection curlightHit;
            curlightHit.HitDistance = glm::distance(ltPos, rayToLight.Origin);
            rayToLight.Direction = toLight;

            if (!Scn->Intersect(rayToLight, curlightHit))
            {
                CalculateInScattering(newRay, toLight, col);
                lightComingIn.AddScaled(col, intensity);
            }
        }

        Color indirect = marchCol;
        Color sum = Color::BLACK;
        int N = NUM_INDIRECT;

        for (int i = 0; i < N; i++)
        {
            Ray rayBounce = CalculateRay(newRay.Origin);
            Intersection bouncehit;
            Color bounceCol = Scn->GetSkyColor();

            if (TraceRay(rayBounce, bouncehit, depth - 1, false))
            {
                bounceCol = bouncehit.Shade;
            }

            CalculateInScattering(newRay, rayBounce.Direction, bounceCol);
            sum.Add(bounceCol);
        }
        sum.Scale(1.f / (float)N);

        indirect.Multiply(sum);
        lightComingIn.Add(indirect);

        CalculateColor(newRay, marchCol, lightComingIn, distToNewRay);

        marchRay = newRay;
    }
}
```

Brightness of directional light is high while the extinction coefficient is ~ 0.9 in order to achieve this effect.