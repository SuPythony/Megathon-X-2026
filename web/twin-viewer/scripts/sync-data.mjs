// Copies the twin's example scenes and appliance visuals from greentwin/twin into src/data,
// so the viewer always renders what the Python scene builder produces. Runs before dev and build.
import { cpSync, mkdirSync, readdirSync, rmSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const twin = join(here, '..', '..', '..', 'greentwin', 'twin')
const out = join(here, '..', 'src', 'data')

rmSync(join(out, 'scenes'), { recursive: true, force: true })
mkdirSync(join(out, 'scenes'), { recursive: true })
for (const f of readdirSync(join(twin, 'examples'))) {
  if (f.endsWith('.scene.json')) cpSync(join(twin, 'examples', f), join(out, 'scenes', f))
}
cpSync(join(twin, 'appliance_visuals.json'), join(out, 'appliance_visuals.json'))
console.log('synced twin scenes and appliance visuals into src/data')
