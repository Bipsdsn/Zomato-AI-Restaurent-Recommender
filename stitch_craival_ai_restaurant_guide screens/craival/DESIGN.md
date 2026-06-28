---
name: Craival
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1b1b1b'
  surface-container: '#20201f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353535'
  on-surface: '#e5e2e1'
  on-surface-variant: '#e4bebc'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#ab8987'
  outline-variant: '#5b403f'
  surface-tint: '#ffb3b1'
  primary: '#ffb3b1'
  on-primary: '#680011'
  primary-container: '#ff535a'
  on-primary-container: '#5b000e'
  inverse-primary: '#bb162c'
  secondary: '#fff9ef'
  on-secondary: '#3a3000'
  secondary-container: '#ffdb3c'
  on-secondary-container: '#725f00'
  tertiary: '#78dc77'
  on-tertiary: '#00390a'
  tertiary-container: '#41a447'
  on-tertiary-container: '#003208'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffdad8'
  primary-fixed-dim: '#ffb3b1'
  on-primary-fixed: '#410007'
  on-primary-fixed-variant: '#92001c'
  secondary-fixed: '#ffe16d'
  secondary-fixed-dim: '#e9c400'
  on-secondary-fixed: '#221b00'
  on-secondary-fixed-variant: '#544600'
  tertiary-fixed: '#94f990'
  tertiary-fixed-dim: '#78dc77'
  on-tertiary-fixed: '#002204'
  on-tertiary-fixed-variant: '#005313'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353535'
typography:
  headline-xl:
    fontFamily: Outfit
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Outfit
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Outfit
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Outfit
    fontSize: 20px
    fontWeight: '500'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  headline-xl-mobile:
    fontFamily: Outfit
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
  margin-mobile: 20px
  margin-desktop: 64px
  gutter: 16px
---

## Brand & Style
The brand personality is confident, appetizing, and effortless. It aims to evoke the feeling of a high-end concierge service—sophisticated yet approachable. This design system uses a **Glassmorphic** style to create depth and focus in a premium dark environment. By layering translucent surfaces over a deep, dark canvas, we create a "night-out" aesthetic that feels modern and polished. The emotional response should be one of hunger-inducing excitement balanced by the calm efficiency of AI-driven curation.

## Colors
The palette is anchored by a vibrant, appetizing **Primary Red (#E23744)** used for calls to action and brand moments. The foundation is a **Deep Charcoal (#1C1C1C)** background, which allows food photography and glass elements to pop. 

- **Primary Red:** High energy, hunger-stimulating. Use for primary buttons and active states.
- **Gold Accent:** Reserved exclusively for star ratings and "Premium" or "Editor's Choice" badges.
- **Success Green:** Used for availability indicators and positive completion states.
- **Glass Surface:** A semi-transparent layer (#2D2D2D at 60-80% opacity) with a background blur (12px-20px) and a 1px white border at 10% opacity.

## Typography
The typography system uses a pairing of **Outfit** for headlines to convey a modern, geometric, and friendly character, while **Inter** provides highly legible utility for body text and UI labels. 

Headlines use tighter letter-spacing to feel impactful and "editorial." Body text uses the Muted color (#A0A0A0) by default to reduce eye strain in dark mode, while headlines and high-priority labels use the Light color (#F4F4F4).

## Layout & Spacing
This design system operates on a rigorous **8px grid**. Layouts should prioritize generous whitespace to maintain a "premium" feel. 

- **Mobile:** Use a single-column fluid layout with 20px side margins. Cards should typically span the full width minus margins.
- **Desktop/Tablet:** Use a 12-column grid. Components like restaurant cards should be grouped in responsive CSS grids (e.g., 3 or 4 columns). 
- **Consistency:** Spacing between related items (e.g., a title and its description) should be 8px (sm). Spacing between sections should be 48px (xxl) to provide clear visual breathing room.

## Elevation & Depth
Depth is communicated through **Glassmorphism** and subtle shadows rather than traditional solid fills. 

- **Level 1 (Background):** The base #1C1C1C surface.
- **Level 2 (Cards/Containers):** Glassmorphic surfaces with a 1px "inner glow" border (white at 10% opacity) and a background blur (saturate 150%, blur 20px). Use a very soft, large-spread shadow (Black at 40% opacity, 20px blur, 10px Y-offset).
- **Level 3 (Modals/Popovers):** Higher transparency contrast and more pronounced shadows to signify immediate interaction priority.

## Shapes
The shape language is rounded and approachable, avoiding sharp corners to maintain a friendly, modern vibe.

- **Cards:** 16px (rounded-lg) for container corners.
- **Input Fields:** 8px (base roundedness) for a structured yet soft feel.
- **Badges/Pills:** 24px (rounded-xl/pill) for category tags, price levels, and status indicators.
- **Buttons:** Match pill-shape (24px+) for primary actions to make them feel more "touchable."

## Components
- **Glassmorphic Cards:** The signature component. Feature a high-quality food image with a gradient overlay (bottom-to-top) to ensure text legibility. The bottom section of the card uses the glass effect.
- **Pill-shaped Badges:** Used for cuisines (e.g., "Italian," "Sushi"). Background should be semi-transparent primary color or glass, with a 1px border.
- **Buttons:** 
  - *Primary:* Solid Red (#E23744) with white text. 
  - *Secondary:* Glass surface with white border and white text.
- **Input Fields:** Semi-transparent dark background with an 8px radius. On **Focus**, the 1px border transitions to Primary Red with a subtle red outer glow.
- **Star Ratings:** Use the Gold (#FFD700) for filled stars. Unfilled stars should be Muted (#A0A0A0) at 30% opacity.
- **AI Recommendation Sparkle:** A custom icon or glow effect using a subtle gradient of Red to Gold to highlight AI-curated "Top Picks."