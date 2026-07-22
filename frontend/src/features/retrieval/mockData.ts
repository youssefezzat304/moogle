export interface RetrievalImage {
  id: string;
  title: string;
  caption: string;
  score: number;
  meta: string;
  crop: string;
}

export interface RetrievalResult {
  id: string;
  title: string;
  region: string;
  lat: number;
  lng: number;
  confidence: number;
  matchText: string;
  summary: string;
  terrain: string;
  lighting: string;
  tags: string[];
  images: RetrievalImage[];
  keywords: string[];
}

export const EVIDENCE_IMAGE_URL = "/lroc_color_16bit_srgb_8k.webp";

export const RETRIEVAL_RESULTS: RetrievalResult[] = [
  {
    id: "apollo-11",
    title: "Apollo 11 landing ellipse",
    region: "Mare Tranquillitatis",
    lat: 0.674,
    lng: 23.473,
    confidence: 0.94,
    matchText: "Basaltic mare, descent-stage vicinity, low-relief regolith",
    summary:
      "Matched orbital frames around the Tranquility Base landing ellipse with smooth mare units and faint ejecta streaking.",
    terrain: "Basalt plain",
    lighting:
      "Key light is slewed onto the near-side mare so the retrieval area stays readable.",
    tags: ["apollo", "mare", "landing site"],
    keywords: [
      "apollo 11",
      "tranquility",
      "tranquillitatis",
      "landing",
      "eagle",
    ],
    images: [
      {
        id: "a11-nac-01",
        title: "NAC strip M175124932",
        caption: "Fine regolith texture and descent-track-like albedo marks.",
        score: 0.92,
        meta: "0.5 m/px · LROC NAC",
        crop: "58% 49%",
      },
      {
        id: "a11-wac-02",
        title: "WAC context mosaic",
        caption: "Low relief mare terrain around the landing ellipse.",
        score: 0.88,
        meta: "100 m/px · WAC",
        crop: "62% 52%",
      },
      {
        id: "a11-dem-03",
        title: "LDEM slope patch",
        caption: "Subtle wrinkle ridge gradients south of the target area.",
        score: 0.81,
        meta: "DEM fused tile",
        crop: "55% 44%",
      },
    ],
  },
  {
    id: "tycho",
    title: "Tycho central peak complex",
    region: "Southern Highlands",
    lat: -43.192,
    lng: -11.36,
    confidence: 0.91,
    matchText: "Fresh rayed crater, high-albedo ejecta, terraced walls",
    summary:
      "Retrieved high-contrast frames over Tycho's peak ring, terraced wall shadows, and radial ejecta structure.",
    terrain: "Complex crater",
    lighting:
      "The virtual sun tracks with the camera to preserve wall detail as the view crosses the terminator.",
    tags: ["crater", "ejecta", "highlands"],
    keywords: ["tycho", "central peak", "rayed crater", "ejecta", "south"],
    images: [
      {
        id: "tycho-nac-01",
        title: "Central peak oblique",
        caption: "Blocky summit texture with steep local shadowing.",
        score: 0.9,
        meta: "NAC pair · stereo candidate",
        crop: "47% 70%",
      },
      {
        id: "tycho-wac-02",
        title: "Ray system context",
        caption: "Bright ejecta fan extending across highland units.",
        score: 0.86,
        meta: "WAC global mosaic",
        crop: "51% 72%",
      },
      {
        id: "tycho-dem-03",
        title: "Terrace elevation tile",
        caption: "Sharp elevation delta from rim to floor.",
        score: 0.79,
        meta: "LDEM slope",
        crop: "43% 66%",
      },
    ],
  },
  {
    id: "shackleton",
    title: "Shackleton rim illumination",
    region: "South Pole",
    lat: -89.9,
    lng: 0,
    confidence: 0.89,
    matchText: "Polar rim, persistent shadow, volatile prospecting context",
    summary:
      "Polar matches emphasize bright rim segments beside permanently shadowed terrain near the south pole.",
    terrain: "Polar crater rim",
    lighting:
      "Scene illumination pivots down to the polar target, keeping the rim visible while the far limb falls dark.",
    tags: ["south pole", "shadow", "volatiles"],
    keywords: [
      "shackleton",
      "south pole",
      "polar",
      "ice",
      "shadow",
      "volatile",
    ],
    images: [
      {
        id: "shack-nac-01",
        title: "Rim glint segment",
        caption: "Illuminated ridge beside a deep shadow field.",
        score: 0.88,
        meta: "Polar NAC tile",
        crop: "50% 96%",
      },
      {
        id: "shack-illum-02",
        title: "Illumination persistence",
        caption: "Mock persistence layer highlighting repeat-lit terrain.",
        score: 0.84,
        meta: "Temporal stack",
        crop: "54% 92%",
      },
      {
        id: "shack-dem-03",
        title: "Rim elevation tile",
        caption: "Steep relief along crater wall and saddle points.",
        score: 0.8,
        meta: "LDEM polar",
        crop: "45% 98%",
      },
    ],
  },
  {
    id: "reiner-gamma",
    title: "Reiner Gamma swirl",
    region: "Oceanus Procellarum",
    lat: 7.5,
    lng: -59,
    confidence: 0.87,
    matchText: "High-albedo lunar swirl, magnetic anomaly, low relief",
    summary:
      "The mock vector search found sinuous albedo patterns around Reiner Gamma with minimal elevation expression.",
    terrain: "Lunar swirl",
    lighting:
      "Light follows the western near-side target so the swirl remains bright instead of sliding into darkness.",
    tags: ["swirl", "albedo", "magnetic"],
    keywords: ["reiner", "gamma", "swirl", "magnetic", "albedo", "procellarum"],
    images: [
      {
        id: "rg-wac-01",
        title: "Swirl albedo ribbon",
        caption: "Bright filamentary pattern against darker mare basalt.",
        score: 0.87,
        meta: "WAC albedo",
        crop: "34% 45%",
      },
      {
        id: "rg-nac-02",
        title: "Mare texture detail",
        caption: "Subtle surface roughness with limited topographic relief.",
        score: 0.82,
        meta: "NAC frame",
        crop: "31% 47%",
      },
      {
        id: "rg-model-03",
        title: "Anomaly overlay",
        caption: "Mock magnetic anomaly footprint aligned with swirl arms.",
        score: 0.78,
        meta: "Vector overlay",
        crop: "37% 42%",
      },
    ],
  },
  {
    id: "aristarchus",
    title: "Aristarchus plateau",
    region: "Northwest Near Side",
    lat: 23.7,
    lng: -47.4,
    confidence: 0.9,
    matchText: "Bright crater, volcanic plateau, sinuous rille candidate",
    summary:
      "Retrieved bright ejecta, plateau boundaries, and nearby rille context from the Aristarchus region.",
    terrain: "Volcanic plateau",
    lighting:
      "The light rig tracks northwest with the camera to keep high-albedo crater walls out of the dark side.",
    tags: ["volcanic", "rille", "plateau"],
    keywords: ["aristarchus", "plateau", "rille", "volcanic", "bright crater"],
    images: [
      {
        id: "aris-nac-01",
        title: "Plateau boundary",
        caption: "Abrupt albedo change along the plateau edge.",
        score: 0.89,
        meta: "NAC detail",
        crop: "36% 36%",
      },
      {
        id: "aris-wac-02",
        title: "Crater ejecta context",
        caption: "Bright ejecta signature on darker plateau material.",
        score: 0.85,
        meta: "WAC mosaic",
        crop: "39% 39%",
      },
      {
        id: "aris-dem-03",
        title: "Rille-adjacent slope",
        caption: "Mock terrain profile near sinuous channel features.",
        score: 0.8,
        meta: "LDEM profile",
        crop: "33% 33%",
      },
    ],
  },
  {
    id: "plato",
    title: "Plato crater floor",
    region: "Mare Imbrium Border",
    lat: 51.6,
    lng: -9.3,
    confidence: 0.86,
    matchText: "Dark flooded crater floor, rim massif, northern mare boundary",
    summary:
      "Matched the smooth, dark floor of Plato and the high-relief rim against northern Imbrium context.",
    terrain: "Flooded crater",
    lighting:
      "The camera and light climb north together so the crater floor is illuminated while the opposite hemisphere dims.",
    tags: ["crater floor", "mare", "rim"],
    keywords: ["plato", "imbrium", "dark floor", "northern", "crater"],
    images: [
      {
        id: "plato-nac-01",
        title: "Floor texture",
        caption: "Smooth mare fill with small craterlets.",
        score: 0.86,
        meta: "NAC frame",
        crop: "48% 21%",
      },
      {
        id: "plato-wac-02",
        title: "Rim massif context",
        caption: "Rugged rim material around low-albedo floor.",
        score: 0.83,
        meta: "WAC mosaic",
        crop: "45% 24%",
      },
      {
        id: "plato-dem-03",
        title: "Rim shadow relief",
        caption: "Elevation falloff from rim crest into the basin.",
        score: 0.77,
        meta: "LDEM tile",
        crop: "51% 19%",
      },
    ],
  },
];

export function resolveMockRetrieval(query: string, fallbackIndex = 0) {
  const normalized = query.toLowerCase();
  const directMatch = RETRIEVAL_RESULTS.find((result) =>
    result.keywords.some((keyword) => normalized.includes(keyword)),
  );

  if (directMatch) {
    return directMatch;
  }

  const hash = Array.from(normalized).reduce(
    (acc, char) => acc + char.charCodeAt(0),
    fallbackIndex,
  );

  return RETRIEVAL_RESULTS[hash % RETRIEVAL_RESULTS.length];
}

export function formatCoords(lat: number, lng: number) {
  const latDir = lat >= 0 ? "N" : "S";
  const lngDir = lng >= 0 ? "E" : "W";

  return `${Math.abs(lat).toFixed(2)} deg ${latDir} / ${Math.abs(lng).toFixed(
    2,
  )} deg ${lngDir}`;
}
