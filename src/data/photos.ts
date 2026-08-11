import type { ImageMetadata } from 'astro';

import seattleDusk from '../assets/photos/DSC00600.jpg';
import seattleMoon from '../assets/photos/DSC00705_moon.jpg';
import yosemiteTunnelView from '../assets/photos/DSC_0625.jpg';
import yosemiteCreek from '../assets/photos/DSC_0466.jpg';
import horseshoeBend from '../assets/photos/DSC_0123.jpg';
import antelopeCanyon from '../assets/photos/DSC_0445.jpg';
import rainierRoad from '../assets/photos/DSC05664.jpg';
import blueAngelsSolo from '../assets/photos/DSC_0268-2.jpg';
import blueAngelsDiamond from '../assets/photos/DSC_0284-2.jpg';
import redArrows from '../assets/photos/DSC_0988.jpg';
import motocross from '../assets/photos/DSC_0024.jpg';
import sunflower from '../assets/photos/DSC05021.jpg';

export interface Photo {
  src: ImageMetadata;
  alt: string;
}

export const photos: Photo[] = [
  {
    src: seattleDusk,
    alt: 'Seattle skyline at dusk from Kerry Park, with the Space Needle in front of a pink sky and Mount Rainier on the horizon',
  },
  {
    src: seattleMoon,
    alt: 'Full moon rising over the Seattle skyline at blue hour, city lights on and Mount Rainier in the distance',
  },
  {
    src: yosemiteTunnelView,
    alt: 'Yosemite Valley from Tunnel View: El Capitan, Half Dome, and Bridalveil Fall above a fog-filled valley floor',
  },
  {
    src: yosemiteCreek,
    alt: 'Long-exposure creek flowing over granite boulders through a mossy evergreen forest',
  },
  {
    src: horseshoeBend,
    alt: 'Horseshoe Bend, where the Colorado River wraps around red sandstone cliffs near Page, Arizona',
  },
  {
    src: antelopeCanyon,
    alt: 'Sunlight falling through the sculpted red sandstone curves of Antelope Canyon',
  },
  {
    src: rainierRoad,
    alt: 'Street lined with evergreens at dusk leading straight toward a towering snow-covered Mount Rainier',
  },
  {
    src: blueAngelsSolo,
    alt: 'US Navy Blue Angels F/A-18 in a high-speed pass, vapor streaming over its wings against a deep blue sky',
  },
  {
    src: blueAngelsDiamond,
    alt: 'Four Blue Angels jets flying in tight diamond formation with smoke trails',
  },
  {
    src: redArrows,
    alt: 'Seven Red Arrows jets climbing in formation, trailing red, white, and blue smoke',
  },
  {
    src: motocross,
    alt: 'Freestyle motocross rider performing an inverted trick high above the clouds',
  },
  {
    src: sunflower,
    alt: 'Honeybee on a sunflower in full bloom, with a field of sunflowers blurred behind',
  },
];
