// Downscale photos for the /photos/ gallery.
//
// Usage: node scripts/resize-photos.mjs <input-dir-or-file> [more inputs...]
// Writes web-friendly JPEGs (max 2000px long edge, quality 82) into
// src/assets/photos/, lowercasing the extension. Run this on new full-size
// photos before adding them to src/data/photos.ts.
import fs from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';

const outDir = 'src/assets/photos';
const inputs = process.argv.slice(2);

if (inputs.length === 0) {
  console.error('Usage: node scripts/resize-photos.mjs <input-dir-or-file> [...]');
  process.exit(1);
}

const files = inputs.flatMap((input) =>
  fs.statSync(input).isDirectory()
    ? fs
        .readdirSync(input)
        .filter((f) => /\.(jpe?g|png)$/i.test(f))
        .map((f) => path.join(input, f))
    : [input]
);

fs.mkdirSync(outDir, { recursive: true });

for (const file of files) {
  const name = path.basename(file).replace(/\.(jpe?g|png)$/i, '.jpg');
  const out = path.join(outDir, name);
  const info = await sharp(file)
    .rotate() // apply EXIF orientation before metadata is stripped
    .resize({ width: 2000, height: 2000, fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 82, mozjpeg: true })
    .toFile(out);
  console.log(`${out}: ${info.width}x${info.height} ${(info.size / 1024).toFixed(0)}KB`);
}
