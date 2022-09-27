import argparse

import image_cropper
import palettepacker

from PIL import Image, ImageOps

HEX_DIGITS = '0123456789abcdef'
BASE32_DIGITS = HEX_DIGITS + 'ghijklmnopqrstuv'


class PaletteError(Exception):
    """Exception class for palette related errors"""


class EbPalette:
    """Represents a color palette of 96 colors (6 subpalettes of 15 colors)"""

    def __init__(self, backdrop=(0, 248, 0)):
        self.backdrop = backdrop
        self.subpalettes = [[] for _ in range(6)]

    def to_image(self):
        """Returns an image representation of the palette"""
        im = Image.new('RGB', (16, 6), self.backdrop)
        for y, colors in enumerate(self.subpalettes):
            for x in range(15):
                try:
                    value = colors[x]
                except IndexError:
                    value = self.backdrop

                im.putpixel((x+1, y), value)

        # Now resize it to 8x size
        im = im.resize((16*8, 6*8), resample=Image.NEAREST)
        return im

    def from_image(self, image):
        """Loads the palette from an image representation"""
        def get_color(im):
            colors = im.getcolors(1)
            assert colors, "Palette image must be 8x8 blocks of solid color"
            return colors[0][1]
        assert image.width == 16*8
        assert image.height == 6*8
        color_images = image_cropper.get_tiles(image, tile_size=8)
        self.subpalettes = [[] for _ in range(6)]
        for index, im_color in enumerate(color_images):
            ypos = index // 16
            xpos = index % 16
            if xpos == 0:
                continue
            color = get_color(im_color)
            self.subpalettes[ypos].append(color)

    def fts_string(self, area_id):
        """Returns the .fts string representation of the palette"""
        palette_str = BASE32_DIGITS[area_id] + '0' # Area, palette 0
        for colors in self.subpalettes:
            palette_str += '000' # Color 0 is always transparent, fixed to 000

            for i in range(15):
                try:
                    r, g, b = colors[i]
                except IndexError:
                    r, g, b = self.backdrop

                palette_str += BASE32_DIGITS[r // 8]
                palette_str += BASE32_DIGITS[g // 8]
                palette_str += BASE32_DIGITS[b // 8]

        return palette_str


class EbTile:
    """Represents an 8x8 tile"""

    def __init__(self, data, palette, palette_row, index=0, is_flipped_h=False, is_flipped_v=False):
        self.data = data
        self.palette = palette
        self.palette_row = palette_row
        self.index = index
        self.is_flipped_h = is_flipped_h
        self.is_flipped_v = is_flipped_v

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                (self.data, self.palette, self.palette_row, self.index, self.is_flipped_h, self.is_flipped_v) ==
                (other.data, other.palette, other.palette_row, self.index, self.is_flipped_h, self.is_flipped_v))

    def __hash__(self):
        return hash((self.data, self.palette, self.palette_row, self.index, self.is_flipped_h, self.is_flipped_v))

    @property
    def is_flipped(self):
        """Returns True if the tile is flipped either horizontally or vertically"""
        return self.is_flipped_h or self.is_flipped_v

    @property
    def is_flipped_hv(self):
        """Returns True if the tile is flipped both horizontally and vertically"""
        return self.is_flipped_h and self.is_flipped_v

    def to_image(self):
        """Returns an image representation of the tile"""
        image = Image.new('RGB', (8, 8))
        colors = self.palette[self.palette_row]
        for y, row in enumerate(self.data):
            for x, pixel in enumerate(row):
                image.putpixel((x, y), colors[pixel])

        return image

    def fts_string(self):
        """Returns the .fts string representation of the tile"""
        return ''.join(HEX_DIGITS[pixel] for pixel in self.data)


class EbChunk:
    """Represents a 32x32 chunk of 16 tiles with surface flag data"""

    def __init__(self, tiles, surface_flags):
        self.tiles = tiles
        self.surface_flags = surface_flags

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                (self.tiles, self.surface_flags) == (other.tiles, other.surface_flags))

    def to_image(self):
        """Returns an image representation of the chunk"""
        image = Image.new('RGB', (32, 32))

        x = 0
        y = 0
        for tile in self.tiles:
            im_tile = tile.to_image()
            image.paste(im_tile, (x, y))

            if x < 32 - 8:
                x += 8
            else:
                x = 0
                y += 8

        return image

    def fts_string(self):
        """Returns the .fts file string representation of the chunk"""
        s = ''
        for i, tile in enumerate(self.tiles):
            snes_tile = tile.index | (0x0800 + (tile.palette_row << 10))
            if tile.is_flipped_h:
                snes_tile |= 0x4000

            if tile.is_flipped_v:
                snes_tile |= 0x8000

            surface = self.surface_flags[i]
            s += f'{snes_tile:04x}{surface:02x}'

        return s


class EbTileset:
    """Represents a collection of unique 32x32 chunks and 8x8 tiles with a palette"""

    def __init__(self, tileset_id):
        if 0 > tileset_id > 19:
            raise ValueError('Tileset ID must be in range 0..19')

        # Data set prior to calling compute()
        self.chunk_tile_images = []
        self.tile_palettes = []
        self.tile_positions = []
        self.map_values = None
        self.map_width = None
        self.palette_provided = False

        # Data set by compute()
        self.tile_index = 0 # Index for next unique tile
        self.tileset_id = tileset_id
        self.chunks = []
        self.tiles = []
        self.palette = EbPalette()
        self.tile_dict = dict()
        self._chunk_image_cache = dict()

    def load_palette(self, image):
        self.palette.from_image(image)
        self.palette_provided = True

    def append_from_image(self, image):
        """Adds unique chunks and tiles from an image into the tileset"""
        chunk_images = image_cropper.get_tiles(image, tile_size=32)
        # Equivalent of ceil(image.width / 32)
        self.map_width = (image.width + 31) // 32
        self.map_values = []

        for chunk_idx, im_chunk in enumerate(chunk_images):
            new_chunk_image = False
            im_chunk_bytes = im_chunk.tobytes()
            if im_chunk_bytes not in self._chunk_image_cache:
                chunk_map_value = len(self._chunk_image_cache)
                self._chunk_image_cache[im_chunk_bytes] = chunk_map_value
                new_chunk_image = True
            self.map_values.append(self._chunk_image_cache[im_chunk_bytes])
            if not new_chunk_image and not self.palette_provided:
                # When we've provided a palette, we need to have accurate
                continue

            tile_images = image_cropper.get_tiles(im_chunk, tile_size=8)
            self.chunk_tile_images.append(tile_images)
            for tile_in_chunk, im_tile in enumerate(tile_images):
                colors = im_tile.getcolors(15) # (count, (r,g,b))
                if colors is None:
                    raise PaletteError('A single tile had more than 15 colors.')

                colors = [rgb for _, rgb in colors] # Discard pixel count
                self.tile_palettes.append(colors)
                tile_x = (chunk_idx %  self.map_width) * 32 + (tile_in_chunk %  4) * 8
                tile_y = (chunk_idx // self.map_width) * 32 + (tile_in_chunk // 4) * 8
                self.tile_positions.append((tile_x, tile_y))

    def compute(self):
        if self.palette_provided:
            subpalette_map = []
            our_palettes = [frozenset(pal) for pal in self.palette.subpalettes]
            for tile_pal, tile_pos in zip(self.tile_palettes, self.tile_positions):
                tile_pal_set = frozenset(tile_pal)
                for pal_idx, potential_pal in enumerate(our_palettes):
                    if tile_pal_set.issubset(potential_pal):
                        subpalette_map.append(pal_idx)
                        break
                else:
                    tile_x, tile_y = tile_pos
                    raise PaletteError("Palette file is not valid for tile at ({},{})".format(tile_x, tile_y))
        else:
            # Use palettepacker library to perform better packing of
            # palettes into subpalettes
            packedSubpalettes, subpalette_map = \
                palettepacker.tilePalettesToSubpalettes(self.tile_palettes)
            assert(len(packedSubpalettes) <= 6)
            # Keep the length of subpalettes the same
            self.palette.subpalettes[:len(packedSubpalettes)] = packedSubpalettes

        for chunk_idx, tile_images in enumerate(self.chunk_tile_images):
            chunk_tiles = []
            for tile_idx, im_tile in enumerate(tile_images):
                im_tile_h = im_tile.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                im_tile_v = im_tile.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                im_tile_hv = im_tile_h.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                tile_hash = im_tile.tobytes()
                tile_h_hash = im_tile_h.tobytes()
                tile_v_hash = im_tile_v.tobytes()
                tile_hv_hash = im_tile_hv.tobytes()
                if tile_hash not in self.tile_dict:
                    palette_row = subpalette_map[chunk_idx * 16 + tile_idx]
                    subpalette = self.palette.subpalettes[palette_row]
                    image_data = im_tile.getdata()
                    tile_data = tuple(subpalette.index(c)+1 for c in image_data)

                    tile = EbTile(tile_data, self.palette, palette_row, index=self.tile_index)
                    tile_h = EbTile(tile_data, self.palette, palette_row, index=self.tile_index, is_flipped_h=True)
                    tile_v = EbTile(tile_data, self.palette, palette_row, index=self.tile_index, is_flipped_v=True)
                    tile_hv = EbTile(tile_data, self.palette, palette_row, index=self.tile_index, is_flipped_h=True, is_flipped_v=True)
                    self.tile_index += 1

                    self.tile_dict[tile_hash] = tile
                    self.tile_dict[tile_h_hash] = tile_h
                    self.tile_dict[tile_v_hash] = tile_v
                    self.tile_dict[tile_hv_hash] = tile_hv
                else:
                    tile = self.tile_dict[tile_hash]

                if tile not in self.tiles:
                    self.tiles.append(tile)

                chunk_tiles.append(tile)

            chunk = EbChunk(chunk_tiles, [0x00] * 16) # Default surface flags to zeros for now...
            if chunk not in self.chunks:
                self.chunks.append(chunk)

    def to_fts(self, filepath):
        """Writes a .fts file containing the data for the tileset"""
        if len(self.tiles) >= 512:
            # TODO Custom exception class
            raise RuntimeError('Oops too many unique tiles!')

        if len(self.chunks) >= 1024:
            # TODO Custom exception class
            raise RuntimeError('Oops too many unique chunks!')

        with open(filepath, 'w', encoding='utf-8', newline='\n') as fts_file:
            # First, 512 tile graphics definition:
            # All 64 pixels for the BACKGROUND tile. Each digit is an index into palette (0..F)
            # All 64 pixels for the FOREGROUND tile. Each digit is an index into palette (0..F)
            # (newline here)

            blank_graphics_str = '0' * 64
            for tile in self.tiles:
                fts_file.write(f'{tile.fts_string()}\n')
                fts_file.write(f'{blank_graphics_str}\n\n') # FOREGROUND is kept blank for now

            for i in range(len(self.tiles), 512):
                fts_file.write(f'{blank_graphics_str}\n')
                fts_file.write(f'{blank_graphics_str}\n\n')

            # Then, a newline followed by the palette information
            # AP(ppp(x16)(x6)), where:
            #   A = "area" (0..31)
            #   P = "palette" (0..7?)
            #   ppp = a single color in base32 (6 subpalettes of 16 colors each)
            palette_str = self.palette.fts_string(self.tileset_id)
            fts_file.write('\n')
            fts_file.write(f'{palette_str}\n')

            # Then, two newlines followed by 1024 "32x32 chunk" definitions:
            # ttttss(x16), where:
            #   t = tile (SNES format)
            #   s = surface flags.
            #   Note: Inexistant chunks use "000000"
            fts_file.write('\n\n')
            for chunk in self.chunks:
                fts_file.write(f'{chunk.fts_string()}\n')

            blank_chunk_str = '0' * 6*16
            for i in range(len(self.chunks), 1024):
                fts_file.write(f'{blank_chunk_str}\n')

    def to_map(self, filepath):
        with open(filepath, 'w', encoding='utf-8', newline='\n') as map_file:
            map_line = []
            for val in self.map_values:
                map_line.append(val)
                if len(map_line) == self.map_width:
                    print(' '.join('{:03x}'.format(x) for x in map_line), file=map_file)
                    map_line.clear()
            assert not map_line, "Internal coding error"

def main(args):
    if len(args.input_files) > 1 and args.output_map:
        print('WARNING: Multiple input files have been specified.')
        print('Only the map for the last input file will be saved.')
    tileset = EbTileset(args.tileset_id)

    if args.palette:
        with Image.open(args.palette) as image:
            image = image.convert(mode='RGB') # Get rid of the alpha channel
            image = ImageOps.posterize(image, 5) # 5-bit color

            tileset.load_palette(image)

    for path in args.input_files:
        with Image.open(path) as image:
            image = image.convert(mode='RGB') # Get rid of the alpha channel
            image = ImageOps.posterize(image, 5) # 5-bit color

            tileset.append_from_image(image)

    tileset.compute()

    print('Done!')
    print(f'{len(tileset.chunks)} chunks!')
    print(f'{len(tileset.tiles)} unique tiles!')

    if args.output_palette:
        print('Writing palette...')
        palette_image = tileset.palette.to_image()
        palette_image.save(args.output_palette)

    print('Writing FTS...')
    tileset.to_fts(args.output)

    if args.output_map:
        print('Writing map...')
        tileset.to_map(args.output_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert image files into CoilSnake-compatible files for Earthbound hacking')

    parser.add_argument('input_files', metavar='IN_IMAGE', nargs='+', help='Input image files')
    parser.add_argument('-t', '--tileset-id', required=True, type=int, metavar='[0-19]', choices=range(20), help='Specify the tileset ID')
    parser.add_argument('-o', '--output', required=True, help='Output FTS file')
    parser.add_argument('-m', '--output-map', help='Output .map file containing chunk indices')
    parser.add_argument('-p', '--palette', help='Palette image file to use as the tileset\'s palette')
    parser.add_argument('--output-palette', help='Output palette as image file')
    args = parser.parse_args()

    from time import perf_counter

    start_time = perf_counter()

    main(args)

    elapsed = perf_counter() - start_time
    print(f'Time taken: {elapsed:.02f}s')
