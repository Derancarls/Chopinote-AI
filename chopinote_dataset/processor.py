"""
MusicXML预处理管道
处理大量MusicXML文件，转换为REMI序列
"""
import os
import json
import pickle
import copy
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

from music21 import converter
import numpy as np
from tqdm import tqdm

try:
    from .converter import MusicXMLToREMI, PDMXToREMI, MIDIToREMI
    from .tokenizer import REMITokenizer
except ImportError:
    from converter import MusicXMLToREMI, PDMXToREMI
    from tokenizer import REMITokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 调号移调：音名 → 音高级别映射
_KEY_PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
           'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
           'A#': 10, 'Bb': 10, 'B': 11}
_KEY_PC_REV = {0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E', 5: 'F',
               6: 'F#', 7: 'G', 8: 'Ab', 9: 'A', 10: 'Bb', 11: 'B'}


def _transpose_key_name(key_sig: str, semitones: int) -> str:
    """将调号名移调 semitones 个半音。"""
    if not key_sig or key_sig == 'unknown':
        return key_sig
    is_minor = key_sig.endswith('m')
    root = key_sig[:-1] if is_minor else key_sig
    new_pc = (_KEY_PC.get(root, 0) + semitones) % 12
    return _KEY_PC_REV[new_pc] + ('m' if is_minor else '')


@dataclass
class MusicMetadata:
    """音乐文件元数据"""
    file_id: str
    file_path: str
    composer: str
    title: str
    genre: str
    year: Optional[int]
    duration_seconds: float
    num_measures: int
    num_notes: int
    num_tokens: int
    time_signature: str
    key_signature: str
    tempo: Optional[float]
    instruments: List[str]
    has_chords: bool
    has_polyphony: bool
    processing_time: float
    hash_md5: str

class _BasePreprocessor:
    """Preprocessor 基类：提供共享的目录/缓存/统计/质量检查方法。"""

    def __init__(self, config_path: str):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        remi_cfg = self.config['dataset']['preprocessing']['remi']
        self.tokenizer = REMITokenizer(
            grid_size=remi_cfg['grid_size'],
            velocity_levels=remi_cfg['velocity_levels'],
        )
        self.cache: dict = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._create_directories()

    def _create_directories(self):
        dirs = [
            self.config['dataset']['storage']['processed_dir'],
            self.config['dataset']['storage']['cache_dir'],
            self.config['dataset']['storage']['token_dir'],
            self.config['dataset']['storage']['metadata_dir'],
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _passes_quality_check(self, metadata: MusicMetadata) -> bool:
        qc = self.config['dataset']['quality_checks']
        fsize = os.path.getsize(metadata.file_path) / 1024
        if fsize < qc['min_file_size_kb'] or fsize > qc['max_file_size_mb'] * 1024:
            return False
        pp = self.config['dataset']['preprocessing']
        if metadata.num_notes < pp['min_notes_per_file'] or metadata.num_notes > pp['max_notes_per_file']:
            return False
        return True

    def _check_sequence_length(self, tokens: List[int]) -> bool:
        pp = self.config['dataset']['preprocessing']
        return pp['min_tokens_per_sequence'] <= len(tokens) <= pp['max_tokens_per_sequence']

    def _save_processed_file(self, file_path: str, tokens: List[int],
                             metadata: MusicMetadata, conversion_metadata: Dict,
                             output_dir: str) -> Dict:
        import time
        t0 = time.time()
        metadata.num_tokens = len(tokens)
        tid = f"{metadata.file_id}.tokens"
        mid = f"{metadata.file_id}.meta.json"
        tp = os.path.join(output_dir, "tokens_v2", tid)
        os.makedirs(os.path.dirname(tp), exist_ok=True)
        with open(tp, 'w', encoding='utf-8') as f:
            json.dump(tokens, f)
        metadata.processing_time = time.time() - t0
        md = asdict(metadata)
        md.update(conversion_metadata)
        mp = os.path.join(output_dir, "metadata_v2", mid)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(md, f, indent=2, ensure_ascii=False)
        return {
            'file_id': metadata.file_id, 'original_path': file_path,
            'token_path': tp, 'metadata_path': mp,
            'num_tokens': len(tokens), 'metadata': md,
        }

    def _compute_file_hash(self, file_path: str) -> str:
        """MD5 hash with .hash sidecar caching (shared across all formats)."""
        sidecar = file_path + '.hash'
        if os.path.exists(sidecar):
            try:
                with open(sidecar) as f:
                    cached = f.read().strip()
                    if cached and len(cached) == 32:
                        return cached
            except (OSError, ValueError):
                pass

        h = self._compute_file_hash_raw(file_path)

        try:
            with open(sidecar, 'w') as f:
                f.write(h)
        except OSError:
            pass

        return h

    @staticmethod
    def _compute_file_hash_raw(file_path: str) -> str:
        """Compute MD5 hash from file content (no sidecar)."""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def _generate_file_id(self, file_path: str) -> str:
        import uuid
        return f"{Path(file_path).stem}_{self._compute_file_hash(file_path)[:8]}_{uuid.uuid4().hex[:8]}"

    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        cf = os.path.join(self.config['dataset']['storage']['cache_dir'], f"{cache_key}.cache")
        if os.path.exists(cf):
            try:
                with open(cf, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def _add_to_cache(self, cache_key: str, data: Dict):
        cf = os.path.join(self.config['dataset']['storage']['cache_dir'], f"{cache_key}.cache")
        try:
            with open(cf, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def _save_processing_stats(self, processed_files: List, failed_files: List, output_dir: str):
        total = len(processed_files) + len(failed_files)
        stats = {
            'total_files': total,
            'processed_files': len(processed_files),
            'failed_files': len(failed_files),
            'success_rate': len(processed_files) / total if total > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (self.cache_hits / (self.cache_hits + self.cache_misses)
                               if (self.cache_hits + self.cache_misses) > 0 else 0),
            'total_tokens': sum(f['num_tokens'] for f in processed_files),
            'avg_tokens_per_file': float(np.mean([f['num_tokens'] for f in processed_files])) if processed_files else 0,
            'composer_distribution': {},
            'genre_distribution': {},
        }
        for f in processed_files:
            c = f['metadata']['composer']
            g = f['metadata']['genre']
            stats['composer_distribution'][c] = stats['composer_distribution'].get(c, 0) + 1
            stats['genre_distribution'][g] = stats['genre_distribution'].get(g, 0) + 1
        sp = os.path.join(output_dir, 'processing_stats.json')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def _infer_genre(self, file_path: str, composer: str = '', title: str = '') -> str:
        """推断音乐体裁（共享实现：文件路径关键词 + 作曲家/标题关键词）。"""
        combined = f'{file_path} {composer} {title}'.lower()
        for kw, genre in [
            ('sonata', 'sonata'), ('ballade', 'ballade'), ('nocturne', 'nocturne'),
            ('etude', 'etude'), ('prelude', 'prelude'), ('fugue', 'fugue'),
            ('chorale', 'chorale'), ('waltz', 'waltz'), ('symphony', 'symphony'),
            ('concerto', 'concerto'), ('mass', 'mass'), ('requiem', 'requiem'),
            ('suite', 'suite'), ('variation', 'variation'), ('rondo', 'rondo'),
            ('march', 'march'), ('polonaise', 'polonaise'), ('mazurka', 'mazurka'),
            ('impromptu', 'impromptu'), ('scherzo', 'scherzo'),
        ]:
            if kw in combined:
                return genre
        return 'unknown'


class MusicXMLPreprocessor(_BasePreprocessor):
    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        remi_cfg = self.config['dataset']['preprocessing']['remi']
        self.converter = MusicXMLToREMI(
            grid_size=remi_cfg['grid_size'],
            velocity_levels=remi_cfg['velocity_levels'],
        )

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """
        处理整个目录的MusicXML文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（如为None则使用配置中的目录）
        """
        if output_dir is None:
            output_dir = self.config['dataset']['storage']['processed_dir']
        
        # 查找所有MusicXML文件
        musicxml_files = self._find_musicxml_files(input_dir)
        logger.info(f"找到 {len(musicxml_files)} 个MusicXML文件")
        
        # 处理每个文件
        processed_files = []
        failed_files = []
        
        for file_path in tqdm(musicxml_files, desc="处理MusicXML文件"):
            try:
                result = self.process_file(file_path, output_dir)
                if result:
                    processed_files.append(result)
                else:
                    failed_files.append(file_path)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                failed_files.append(file_path)
        
        # 保存处理统计
        self._save_processing_stats(processed_files, failed_files, output_dir)
        
        return processed_files, failed_files
    
    def _find_musicxml_files(self, directory: str) -> List[str]:
        """递归查找MusicXML文件"""
        musicxml_extensions = ['.musicxml', '.xml', '.mxl']
        files = []
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in musicxml_extensions):
                    files.append(os.path.join(root, filename))
        
        return files
    
    def process_file(self, file_path: str, output_dir: str) -> Optional[Dict]:
        """
        处理单个MusicXML文件
        
        Args:
            file_path: MusicXML文件路径
            output_dir: 输出目录
        
        Returns:
            处理结果字典，包含元数据和token路径
        """
        # 检查缓存
        file_hash = self._compute_file_hash(file_path)
        cache_key = f"{file_hash}_{self.config['dataset']['preprocessing']['remi']['grid_size']}"
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self.cache_hits += 1
            logger.debug(f"缓存命中: {file_path}")
            return cached_result
        
        self.cache_misses += 1
        
        try:
            # 1. 解析MusicXML
            score = converter.parse(file_path)
            
            # 2. 提取元数据
            metadata = self._extract_metadata(file_path, score)
            
            # 3. 质量检查
            if not self._passes_quality_check(metadata):
                logger.warning(f"文件未通过质量检查: {file_path}")
                return None
            
            # 4. 转换为REMI tokens
            tokens, conversion_metadata = self.converter.convert(file_path, collect_metadata=True)
            
            # 5. 序列长度检查
            if not self._check_sequence_length(tokens):
                logger.warning(f"序列长度不合适: {file_path} ({len(tokens)} tokens)")
                return None
            
            # 6. 保存结果
            result = self._save_processed_file(
                file_path, tokens, metadata, conversion_metadata, output_dir
            )
            
            # 7. 更新缓存
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"处理文件时出错 {file_path}: {e}")
            return None
    
    def _extract_metadata(self, file_path: str, score) -> MusicMetadata:
        """从MusicXML文件中提取元数据"""
        from music21 import metadata, tempo
        
        # 获取文件信息
        file_id = self._generate_file_id(file_path)
        
        # 从music21提取元数据
        md = score.metadata
        
        # 计算音乐特征
        quarter_length = score.duration.quarterLength
        num_measures = len(score.parts[0].getElementsByClass('Measure'))
        
        # 统计音符数量
        num_notes = 0
        for part in score.parts:
            num_notes += len(part.flat.notes)
        
        # 提取乐器信息
        instruments = []
        for part in score.parts:
            if part.partName:
                instruments.append(part.partName)
        
        # 检查多声部
        has_polyphony = len(score.parts) > 1
        
        # 检查和弦
        has_chords = any(len(chord.notes) > 1 for chord in score.flat.getElementsByClass('Chord'))
        
        # 提取拍号和调号
        time_sig = str(score.flat.getElementsByClass('TimeSignature')[0]) if score.flat.getElementsByClass('TimeSignature') else "4/4"
        key_sig = str(score.flat.getElementsByClass('KeySignature')[0]) if score.flat.getElementsByClass('KeySignature') else "C major"
        
        # 提取速度
        tempo_mark = score.flat.getElementsByClass(tempo.MetronomeMark)
        tempo_value = None
        if tempo_mark and tempo_mark[0].number is not None:
            try:
                tempo_value = float(tempo_mark[0].number)
            except (ValueError, TypeError):
                pass
        
        return MusicMetadata(
            file_id=file_id,
            file_path=file_path,
            composer=md.composer if md and md.composer else "Unknown",
            title=md.title if md and md.title else Path(file_path).stem,
            genre=self._infer_genre(file_path, composer=md.composer if md and md.composer else '',
                                 title=md.title if md and md.title else ''),
            year=md.date if md and md.date else None,
            # quarterLength → seconds: 使用速度（无速度时默认 120 BPM）
            duration_seconds=quarter_length * 60.0 / (tempo_value or 120.0),
            num_measures=num_measures,
            num_notes=num_notes,
            num_tokens=0,  # 将在后面更新
            time_signature=time_sig,
            key_signature=key_sig,
            tempo=tempo_value,
            instruments=instruments,
            has_chords=has_chords,
            has_polyphony=has_polyphony,
            processing_time=0.0,  # 将在后面更新
            hash_md5=self._compute_file_hash(file_path)
        )
    

class PDMXPreprocessor(_BasePreprocessor):
    """PDMX JSON → REMI 预处理管道

    直接处理 PDMX 数据集的 JSON 格式（无需 music21 解析），
    输出与 MusicXMLPreprocessor 兼容的 token 和元数据。
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        remi_cfg = self.config['dataset']['preprocessing']['remi']
        self.converter = PDMXToREMI(
            grid_size=remi_cfg['grid_size'],
            velocity_levels=remi_cfg['velocity_levels'],
        )

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """处理 PDMX 目录结构下的所有 JSON 数据文件。"""
        if output_dir is None:
            output_dir = self.config['dataset']['storage']['processed_dir']

        json_files = self._find_pdmx_files(input_dir)
        logger.info(f"找到 {len(json_files)} 个 PDMX JSON 文件")

        processed_files = []
        failed_files = []

        for file_path in tqdm(json_files, desc="处理 PDMX 文件"):
            try:
                result = self.process_file(file_path, output_dir)
                if result:
                    processed_files.append(result)
                else:
                    failed_files.append(file_path)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                failed_files.append(file_path)

        self._save_processing_stats(processed_files, failed_files, output_dir)
        return processed_files, failed_files

    def _find_pdmx_files(self, directory: str) -> List[str]:
        """递归查找 PDMX 数据 JSON（跳过 data/metadata 目录下的纯元数据 JSON）。"""
        files = []
        for root, dirs, fnames in os.walk(directory):
            # 剪枝：不进入 metadata 子目录（纯元数据 JSON，无法转换）
            dirs[:] = [d for d in dirs if d != 'metadata']
            for fname in fnames:
                if fname.endswith('.json'):
                    files.append(os.path.join(root, fname))
        return files

    def process_file(self, file_path: str, output_dir: str) -> Optional[Dict]:
        """处理单个 PDMX JSON 文件。"""
        file_hash = self._compute_file_hash(file_path)
        cache_key = f"pdmx_{file_hash}_{self.config['dataset']['preprocessing']['remi']['grid_size']}"

        cached = self._get_from_cache(cache_key)
        if cached:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        try:
            # 1. 读取 PDMX JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                pdmx_data = json.load(f)

            # 2. 提取元数据
            metadata = self._extract_metadata(file_path, pdmx_data)

            # 3. 质量检查
            if not self._passes_quality_check(metadata):
                logger.warning(f"文件未通过质量检查: {file_path}")
                return None

            # 4. 转换为 REMI tokens
            tokens, conv_meta = self.converter.convert_pdmx(pdmx_data, collect_metadata=True)

            # 5. 序列长度检查
            if not self._check_sequence_length(tokens):
                logger.warning(f"序列长度不合适: {file_path} ({len(tokens)} tokens)")
                return None

            # 6. 保存结果
            result = self._save_processed_file(file_path, tokens, metadata, conv_meta, output_dir)

            # 7. 缓存
            self._add_to_cache(cache_key, result)

            # ── 8. 移调增强 ──────────────────────────────────────
            augment_cfg = self.config.get('dataset', {}).get('preprocessing', {}).get('augment', {})
            if augment_cfg.get('transpose', False):
                tr = augment_cfg.get('transpose_range', 3)
                for semitone in range(-tr, tr + 1):
                    if semitone == 0:
                        continue
                    transposed = self._transpose_pdmx(pdmx_data, semitone)
                    if transposed is None:
                        continue  # 音高越界
                    try:
                        tt, tc = self.converter.convert_pdmx(transposed, collect_metadata=True)
                        if not self._check_sequence_length(tt):
                            continue
                        # 派生元数据（保持原 metadata 大部分字段，只改 file_id/title/key_signature）
                        tm = copy.deepcopy(metadata)
                        sign = '+' if semitone > 0 else ''
                        tm.file_id = f"{metadata.file_id}_t{sign}{semitone}"
                        tm.key_signature = _transpose_key_name(tm.key_signature, semitone)
                        tm.num_tokens = len(tt)
                        self._save_processed_file(file_path, tt, tm, tc, output_dir)
                    except Exception:
                        continue
            # ────────────────────────────────────────────────────────

            return result

        except Exception as e:
            logger.error(f"处理 PDMX 文件时出错 {file_path}: {e}")
            return None

    def _extract_metadata(self, file_path: str, pdmx_data: dict) -> MusicMetadata:
        from pathlib import Path
        md = pdmx_data.get('metadata', {})

        title = md.get('title') or Path(file_path).stem
        creators = md.get('creators', [])
        composer = creators[0] if creators else 'Unknown'

        tracks = pdmx_data.get('tracks', [])
        instruments = [t.get('name', f'Track_{i}') for i, t in enumerate(tracks)]
        num_notes = sum(len(t.get('notes', [])) for t in tracks)
        has_chords = any(
            len([n for n in t.get('notes', []) if n.get('pitch') is not None]) > 1
            for t in tracks
        )

        # 估计小节数和时长
        barlines = pdmx_data.get('barlines', [])
        num_measures = len(barlines) if barlines else 1
        song_length = pdmx_data.get('song_length', 0)
        resolution = pdmx_data.get('resolution', 480)

        # 拍号
        ts_list = pdmx_data.get('time_signatures', [])
        time_sig = f'{ts_list[0]["numerator"]}/{ts_list[0]["denominator"]}' if ts_list else '4/4'

        # 调号（从 PDMX key_signatures 字段提取）
        ks_list = pdmx_data.get('key_signatures', [])
        if ks_list:
            ks = ks_list[0]
            root_str = ks.get('root_str', '')
            if not isinstance(root_str, str):
                root_str = ''
            key_sig = root_str + ('m' if ks.get('mode') == 'minor' else '')
        else:
            key_sig = 'unknown'

        # 速度（先提取，用于时长计算）
        tempos = pdmx_data.get('tempos', [])
        tempo_val = float(tempos[0]['qpm']) if tempos else None

        # quarter_notes = song_length / resolution; seconds = quarter_notes * 60 / qpm
        duration_seconds = (song_length / resolution) * 60.0 / (tempo_val or 120.0) if song_length > 0 and resolution > 0 else 0.0

        return MusicMetadata(
            file_id=self._generate_file_id(file_path),
            file_path=file_path,
            composer=composer,
            title=title,
            genre=self._infer_genre(file_path, composer=composer, title=title),
            year=None,
            duration_seconds=duration_seconds,
            num_measures=num_measures,
            num_notes=num_notes,
            num_tokens=0,
            time_signature=time_sig,
            key_signature=key_sig,
            tempo=tempo_val,
            instruments=instruments,
            has_chords=has_chords,
            has_polyphony=len(tracks) > 1,
            processing_time=0.0,
            hash_md5=self._compute_file_hash(file_path),
        )

    # ---- 以下方法与 MusicXMLPreprocessor 相同 ----

    def _transpose_pdmx(self, pdmx_data: dict, semitones: int) -> Optional[dict]:
        """返回移调后的 pdmx_data deepcopy，音高越界则返回 None。"""
        data = copy.deepcopy(pdmx_data)
        for track in data.get('tracks', []):
            for note in track.get('notes', []):
                pitch = note.get('pitch')
                if pitch is not None:
                    new_pitch = pitch + semitones
                    if new_pitch < 0 or new_pitch > 127:
                        return None
                    note['pitch'] = new_pitch
        return data


class MIDIPreprocessor(_BasePreprocessor):
    """MIDI → REMI 预处理管道。

    用 music21 解析 MIDI 文件，通过 MIDIToREMI 转换为 token 序列，
    输出与 MusicXMLPreprocessor / PDMXPreprocessor 兼容的 token 和元数据。
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__(config_path)
        remi_cfg = self.config['dataset']['preprocessing']['remi']
        self.converter = MIDIToREMI(
            grid_size=remi_cfg['grid_size'],
            velocity_levels=remi_cfg['velocity_levels'],
        )

    def process_directory(self, input_dir: str, output_dir: Optional[str] = None):
        """处理目录下所有 MIDI 文件。"""
        if output_dir is None:
            output_dir = self.config['dataset']['storage']['processed_dir']

        midi_files = self._find_midi_files(input_dir)
        logger.info(f"找到 {len(midi_files)} 个 MIDI 文件")

        processed_files = []
        failed_files = []

        for file_path in tqdm(midi_files, desc="处理 MIDI 文件"):
            try:
                result = self.process_file(file_path, output_dir)
                if result:
                    processed_files.append(result)
                else:
                    failed_files.append(file_path)
            except Exception as e:
                logger.error(f"处理文件失败 {file_path}: {e}")
                failed_files.append(file_path)

        self._save_processing_stats(processed_files, failed_files, output_dir)
        return processed_files, failed_files

    def _find_midi_files(self, directory: str) -> List[str]:
        """递归查找 .mid / .midi 文件。"""
        files = []
        for root, _dirs, fnames in os.walk(directory):
            for fname in fnames:
                if fname.lower().endswith(('.mid', '.midi')):
                    files.append(os.path.join(root, fname))
        return files

    def process_file(self, file_path: str, output_dir: str) -> Optional[Dict]:
        """处理单个 MIDI 文件。"""
        file_hash = self._compute_file_hash(file_path)
        cache_key = f"midi_{file_hash}_{self.config['dataset']['preprocessing']['remi']['grid_size']}"

        cached = self._get_from_cache(cache_key)
        if cached:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        try:
            # 1. 解析 MIDI
            score = converter.parse(file_path)

            # 2. 提取元数据
            metadata = self._extract_metadata(file_path, score)

            # 3. 质量检查
            if not self._passes_quality_check(metadata):
                logger.warning(f"文件未通过质量检查: {file_path}")
                return None

            # 4. 转换为 REMI tokens
            tokens, conv_meta = self.converter.convert_score(score, collect_metadata=True)

            # 5. 序列长度检查
            if not self._check_sequence_length(tokens):
                logger.warning(f"序列长度不合适: {file_path} ({len(tokens)} tokens)")
                return None

            # 6. 保存结果
            result = self._save_processed_file(file_path, tokens, metadata, conv_meta, output_dir)

            # 7. 缓存
            self._add_to_cache(cache_key, result)

            # ── 8. 移调增强 ──────────────────────────────────────
            augment_cfg = self.config.get('dataset', {}).get('preprocessing', {}).get('augment', {})
            if augment_cfg.get('transpose', False):
                tr = augment_cfg.get('transpose_range', 3)
                for semitone in range(-tr, tr + 1):
                    if semitone == 0:
                        continue
                    try:
                        ts = score.transpose(semitone, inPlace=False)
                    except Exception:
                        continue
                    # 检查是否有音高越界
                    all_pitches = []
                    for n in ts.flatten().notesAndRests:
                        if hasattr(n, 'pitch') and hasattr(n.pitch, 'midi'):
                            all_pitches.append(n.pitch.midi)
                    if all_pitches and (min(all_pitches) < 0 or max(all_pitches) > 127):
                        continue
                    try:
                        tt, tc = self.converter.convert_score(ts, collect_metadata=True)
                        if not self._check_sequence_length(tt):
                            continue
                        tm = copy.deepcopy(metadata)
                        sign = '+' if semitone > 0 else ''
                        tm.file_id = f"{metadata.file_id}_t{sign}{semitone}"
                        tm.key_signature = _transpose_key_name(tm.key_signature, semitone)
                        tm.num_tokens = len(tt)
                        self._save_processed_file(file_path, tt, tm, tc, output_dir)
                    except Exception:
                        continue
            # ────────────────────────────────────────────────────────

            return result

        except Exception as e:
            logger.error(f"处理 MIDI 文件时出错 {file_path}: {e}")
            return None

    def _extract_metadata(self, file_path: str, score) -> MusicMetadata:
        """从 music21 Score 和文件路径提取元数据。"""
        from music21 import key, tempo

        p = Path(file_path)
        # 作曲家：从路径中推断（如 .../asap/Bach/Fugue/... → Bach）
        path_parts = p.parts
        composer = 'Unknown'
        for i, part in enumerate(path_parts):
            if part.lower() in ASAP_COMPOSERS:
                composer = part
                break
        if composer == 'Unknown' and len(path_parts) >= 3:
            # 倒数第二级目录可能包含作曲家名
            composer = path_parts[-2]

        title = p.stem

        # 音符统计
        all_notes_nr = score.flatten().notesAndRests
        notes_only = [n for n in all_notes_nr if hasattr(n, 'pitch')]
        num_notes = len(notes_only)

        # 小节数
        measures = list(score.flatten().getElementsByClass('Measure'))
        num_measures = len(measures) if measures else 1

        # 拍号
        ts = score.flatten().getElementsByClass('TimeSignature')
        time_sig = f'{ts[0].numerator}/{ts[0].denominator}' if ts else '4/4'

        # 调号
        ks = score.flatten().getElementsByClass(key.Key)
        if ks:
            k = ks[0]
            key_sig = k.tonic.name + ('m' if k.mode == 'minor' else '')
        else:
            key_sig = 'unknown'

        # 速度
        tms = score.flatten().getElementsByClass(tempo.MetronomeMark)
        tempo_val = float(tms[0].number) if tms and tms[0].number else None

        # 乐器
        instruments = []
        for part in score.parts:
            instrs = list(part.flatten().getElementsByClass('Instrument'))
            instruments.append(str(instrs[0]) if instrs else 'Unknown')

        # 时长：quarterLength → seconds（无速度时默认 120 BPM）
        duration_seconds = score.duration.quarterLength * 60.0 / (tempo_val or 120.0)

        return MusicMetadata(
            file_id=self._generate_file_id(file_path),
            file_path=file_path,
            composer=composer,
            title=title,
            genre=self._infer_genre(file_path, composer=composer, title=title),
            year=None,
            duration_seconds=duration_seconds,
            num_measures=num_measures,
            num_notes=num_notes,
            num_tokens=0,
            time_signature=time_sig,
            key_signature=key_sig,
            tempo=tempo_val,
            instruments=instruments,
            has_chords=any(len(ch.notes) > 1 for ch in score.flat.getElementsByClass('Chord')),
            has_polyphony=len(list(score.parts)) > 1,
            processing_time=0.0,
            hash_md5=self._compute_file_hash(file_path),
        )



# ASAP 作曲家目录名（用于元数据推断）
ASAP_COMPOSERS = {
    'bach', 'balakirev', 'beethoven', 'brahms', 'chopin', 'debussy',
    'glinka', 'haydn', 'liszt', 'mozart', 'prokofiev', 'rachmaninoff',
    'ravel', 'schubert', 'schumann', 'scriabin',
}
