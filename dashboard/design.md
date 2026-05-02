---
name: Kinetic Neural OS
colors:
  surface: '#131314'
  surface-dim: '#131314'
  surface-bright: '#3a393a'
  surface-container-lowest: '#0e0e0f'
  surface-container-low: '#1c1b1c'
  surface-container: '#201f20'
  surface-container-high: '#2a2a2b'
  surface-container-highest: '#353436'
  on-surface: '#e5e2e3'
  on-surface-variant: '#c4c9ac'
  inverse-surface: '#e5e2e3'
  inverse-on-surface: '#313031'
  outline: '#8e9379'
  outline-variant: '#444933'
  surface-tint: '#abd600'
  primary: '#ffffff'
  on-primary: '#283500'
  primary-container: '#c3f400'
  on-primary-container: '#556d00'
  inverse-primary: '#506600'
  secondary: '#b9f1ff'
  on-secondary: '#00363f'
  secondary-container: '#00e0ff'
  on-secondary-container: '#005f6d'
  tertiary: '#ffffff'
  on-tertiary: '#452b00'
  tertiary-container: '#ffddb3'
  on-tertiary-container: '#8a5b00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#c3f400'
  primary-fixed-dim: '#abd600'
  on-primary-fixed: '#161e00'
  on-primary-fixed-variant: '#3c4d00'
  secondary-fixed: '#a5eeff'
  secondary-fixed-dim: '#00daf8'
  on-secondary-fixed: '#001f25'
  on-secondary-fixed-variant: '#004e5a'
  tertiary-fixed: '#ffddb3'
  tertiary-fixed-dim: '#ffb950'
  on-tertiary-fixed: '#291800'
  on-tertiary-fixed-variant: '#624000'
  background: '#131314'
  on-background: '#e5e2e3'
  surface-variant: '#353436'
typography:
  display-xl:
    fontFamily: Space Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 32px
    fontWeight: '500'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '500'
    lineHeight: '1.2'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.5'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-caps:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1.0'
    letterSpacing: 0.1em
  data-num:
    fontFamily: Space Grotesk
    fontSize: 20px
    fontWeight: '500'
    lineHeight: '1.0'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin: 32px
  container-padding: 24px
---

## Brand & Style

This design system is engineered for the high-performance automotive sector, targeting drivers who demand instantaneous data clarity and a sophisticated, cockpit-inspired environment. The brand personality is precise, intelligent, and vigilant. It evokes a sense of "calm control" through a minimal, high-tech aesthetic that balances complex telemetry with intuitive interactions.

The visual style is a hybrid of **Glassmorphism** and **Minimalism**. It utilizes deep layered surfaces to create a sense of three-dimensional space within the dashboard. The interface prioritizes functional density—displaying real-time diagnostics and environmental data without overwhelming the user—through the use of hairline strokes, translucent materials, and a focus on essentialism.

## Colors

The palette is anchored in a monochromatic dark spectrum to maximize contrast and reduce night-driving eye fatigue. 

*   **Base (Neutral):** A deep, obsidian charcoal (`#0A0A0B`) serves as the foundation, with lighter grey variants used for elevated glass panels.
*   **Active/Safety (Primary):** Neon Lime Green (`#CCFF00`) is used exclusively for active states, safety indicators, and critical navigational paths. 
*   **Data Visualization:** Subtle Blue (`#00E0FF`) represents fluid systems, climate, and connectivity, while Amber (`#FFAB00`) is reserved for warnings and secondary alerts.
*   **Text:** Pure, crisp white is used for high-readability labels, with reduced-opacity white for metadata.

## Typography

The typography strategy leverages two distinct sans-serif families to balance technical character with readability. **Space Grotesk** is used for headlines, data readouts, and labels to provide a futuristic, geometric edge that feels engineered. **Inter** is utilized for body copy and long-form text to ensure maximum legibility at a glance.

Data density is maintained by using tight line heights for numerical readouts and generous letter spacing for all-caps labels, ensuring that even at small scales, the information is distinct and professional.

## Layout & Spacing

The design system employs a **fluid grid** model designed for wide-format automotive displays. It utilizes a 12-column system that adapts to various module sizes—from full-width navigation maps to narrow vertical widgets for climate control.

The spacing rhythm is based on a **4px base unit**. Elements are grouped within glass "containers" with a standard 16px gutter between them. This allows for high information density while maintaining enough breathing room to prevent the interface from feeling cluttered or distracting during high-speed operation.

## Elevation & Depth

Depth is achieved through a **glassmorphic hierarchy** rather than traditional shadows. 

1.  **Backdrop:** The base layer is the darkest neutral.
2.  **Primary Panels:** Use a subtle background blur (20px–40px) and a low-opacity white tint (5-8%). They are defined by a 1px "hairline" border with a linear gradient (top-left to bottom-right) to simulate a light catch on the edge.
3.  **Active Elements:** Elements requiring immediate attention are "lifted" using a subtle outer glow in the primary lime color, creating a floating effect.
4.  **Interactive Overlays:** Use a higher opacity tint and stronger blur to visually separate modal content from the background telemetry.

## Shapes

The shape language reflects modern industrial design—blending organic curves with technical precision. Most containers use a **rounded** (0.5rem) corner radius to soften the high-tech aesthetic and make the interface feel approachable. 

Iconography and buttons utilize "pill-shaped" geometry for primary actions to distinguish them from informational data cards. Decorative elements, such as progress bars and graph containers, may use sharp inner corners combined with rounded outer shells to emphasize an "engineered" look.

## Components

*   **Buttons:** Primary buttons are pill-shaped with a solid Lime Green fill and black text. Secondary buttons are ghost-style with the hairline border and white text.
*   **Data Visualization:** Real-time gauges (speed, battery, RPM) use segmented arcs with a "glow-trail" effect. The active segment is Lime Green, while the "track" is a low-opacity grey.
*   **Glass Cards:** Informational tiles (Weather, Media, Calendar) use the glassmorphic style. Content within cards is strictly aligned to a sub-grid for high density.
*   **Control Sliders:** Thin horizontal tracks with a circular handle. The active portion of the track glows in the primary color.
*   **Status Chips:** Small, semi-transparent capsules used for connectivity (5G, Bluetooth) or vehicle status (Locked/Unlocked).
*   **Interactive Toggles:** Large, tactile switches with a distinct Lime Green "On" state.
*   **Visual Indicators:** Small animated pulses are used for active voice-AI listening and real-time sensor pings to show the system is "alive."
