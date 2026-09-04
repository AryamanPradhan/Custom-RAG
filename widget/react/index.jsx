/**
 * React wrapper for Next.js / Vite projects.
 *
 * It mounts the same custom element the script tag does, so there is one
 * implementation and one place a bug gets fixed.
 *
 *   import { HotelGuide } from "@hotel-guide/react";
 *   <HotelGuide propertyId="casa-verde" endpoint="https://guide.example.com" />
 *
 * Next.js: this touches customElements, so it must run on the client. Either
 * mark the importing file "use client", or load it with
 *   dynamic(() => import("@hotel-guide/react"), { ssr: false })
 */

import { useEffect, useRef } from "react";

export function HotelGuide({
  propertyId,
  endpoint,
  title,
  greeting,
  accent,
}) {
  const ref = useRef(null);

  useEffect(() => {
    // The custom element registers itself on import, and importing it inside
    // the effect keeps `document` out of the server render path.
    let cancelled = false;
    import("../src/guide.js").then(() => {
      if (cancelled || !ref.current) return;
      if (accent) ref.current.style.setProperty("--guide-accent", accent);
    });
    return () => {
      cancelled = true;
    };
  }, [accent]);

  return (
    <hotel-guide
      ref={ref}
      property-id={propertyId}
      endpoint={endpoint}
      {...(title ? { "title-text": title } : {})}
      {...(greeting ? { greeting } : {})}
    />
  );
}

export default HotelGuide;
