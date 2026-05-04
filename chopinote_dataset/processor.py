"""
MusicXML预处理管道
处理大量MusicXML文件，转换为REMI序列
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

from music21 import converter
import numpy as np
from tqdm import tqdm

try:
    from .converter import MusicXMLToREMI
    from .tokenizer import REMITokenizer
except ImportError:
    from converter import MusicXMLToREMI
    from tokenizer import REMITokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class MusicXMLPreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化预处理管道
        
        Args:
            config_path: 配置文件路径
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化转换器（从配置读取 REMI 参数）
        remi_cfg = self.config['dataset']['preprocessing']['remi']
        self.converter = MusicXMLToREMI(
            grid_size=remi_cfg['grid_size'],
            velocity_levels=remi_cfg['velocity_levels'],
        )
        self.tokenizer = REMITokenizer(
            grid_size=remi_cfg['grid_size'],
            velocity_levels=remi_cfg['velocity_levels'],
        )
        
        # 创建目录
        self._create_directories()
        
        # 缓存
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['dataset']['storage']['processed_dir'],
            self.config['dataset']['storage']['cache_dir'],
            self.config['dataset']['storage']['token_dir'],
            self.config['dataset']['storage']['metadata_dir'],
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
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
        duration = score.duration.quarterLength
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
        tempo_value = float(tempo_mark[0].number) if tempo_mark else None
        
        return MusicMetadata(
            file_id=file_id,
            file_path=file_path,
            composer=md.composer if md and md.composer else "Unknown",
            title=md.title if md and md.title else Path(file_path).stem,
            genre=self._infer_genre(file_path, md),
            year=md.date if md and md.date else None,
            duration_seconds=duration,
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
    
    def _infer_genre(self, file_path: str, md) -> str:
        """推断音乐体裁"""
        # 从文件名和路径推断
        path_lower = file_path.lower()
        
        if 'ballad' in path_lower:
            return 'ballade'
        elif 'nocturne' in path_lower:
            return 'nocturne'
        elif 'etude' in path_lower:
            return 'etude'
        elif 'sonata' in path_lower:
            return 'sonata'
        elif 'prelude' in path_lower:
            return 'prelude'
        elif 'fugue' in path_lower:
            return 'fugue'
        elif 'chorale' in path_lower:
            return 'chorale'
        elif md and md.genericName:
            return md.genericName
        else:
            return 'unknown'
    
    def _passes_quality_check(self, metadata: MusicMetadata) -> bool:
        """质量检查"""
        config = self.config['dataset']['quality_checks']
        
        # 检查文件大小
        file_size_kb = os.path.getsize(metadata.file_path) / 1024
        if file_size_kb < config['min_file_size_kb']:
            logger.warning(f"文件太小: {metadata.file_path} ({file_size_kb:.1f}KB)")
            return False
        
        if file_size_kb > config['max_file_size_mb'] * 1024:
            logger.warning(f"文件太大: {metadata.file_path} ({file_size_kb/1024:.1f}MB)")
            return False
        
        # 检查音符数量
        if metadata.num_notes < self.config['dataset']['preprocessing']['min_notes_per_file']:
            logger.warning(f"音符太少: {metadata.file_path} ({metadata.num_notes} notes)")
            return False
        
        if metadata.num_notes > self.config['dataset']['preprocessing']['max_notes_per_file']:
            logger.warning(f"音符太多: {metadata.file_path} ({metadata.num_notes} notes)")
            return False
        
        return True
    
    def _check_sequence_length(self, tokens: List[int]) -> bool:
        """检查序列长度是否合适"""
        config = self.config['dataset']['preprocessing']
        
        if len(tokens) < config['min_tokens_per_sequence']:
            return False
        
        if len(tokens) > config['max_tokens_per_sequence']:
            return False
        
        return True
    
    def _save_processed_file(self, file_path: str, tokens: List[int], 
                           metadata: MusicMetadata, conversion_metadata: Dict,
                           output_dir: str) -> Dict:
        """保存处理后的文件"""
        import time
        start_time = time.time()
        
        # 更新元数据
        metadata.num_tokens = len(tokens)
        
        # 生成输出文件名
        base_name = Path(file_path).stem
        token_filename = f"{metadata.file_id}.tokens"
        metadata_filename = f"{metadata.file_id}.meta.json"
        
        # 保存tokens
        token_path = os.path.join(output_dir, "tokens", token_filename)
        with open(token_path, 'w', encoding='utf-8') as f:
            json.dump(tokens, f)
        
        # 保存元数据
        metadata.processing_time = time.time() - start_time
        metadata_dict = asdict(metadata)
        metadata_dict.update(conversion_metadata)
        
        metadata_path = os.path.join(output_dir, "metadata", metadata_filename)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        # 返回结果
        result = {
            'file_id': metadata.file_id,
            'original_path': file_path,
            'token_path': token_path,
            'metadata_path': metadata_path,
            'num_tokens': len(tokens),
            'metadata': metadata_dict
        }
        
        return result
    
    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _generate_file_id(self, file_path: str) -> str:
        """生成唯一的文件ID"""
        import uuid
        file_hash = self._compute_file_hash(file_path)
        return f"{Path(file_path).stem}_{file_hash[:8]}_{uuid.uuid4().hex[:8]}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取数据"""
        cache_file = os.path.join(
            self.config['dataset']['storage']['cache_dir'],
            f"{cache_key}.cache"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _add_to_cache(self, cache_key: str, data: Dict):
        """添加数据到缓存"""
        cache_file = os.path.join(
            self.config['dataset']['storage']['cache_dir'],
            f"{cache_key}.cache"
        )
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def _save_processing_stats(self, processed_files: List, failed_files: List, output_dir: str):
        """保存处理统计"""
        stats = {
            'total_files': len(processed_files) + len(failed_files),
            'processed_files': len(processed_files),
            'failed_files': len(failed_files),
            'success_rate': len(processed_files) / (len(processed_files) + len(failed_files)) if (len(processed_files) + len(failed_files)) > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_tokens': sum(f['num_tokens'] for f in processed_files),
            'avg_tokens_per_file': np.mean([f['num_tokens'] for f in processed_files]) if processed_files else 0,
            'composer_distribution': {},
            'genre_distribution': {},
        }
        
        # 统计作曲家和体裁分布
        for file in processed_files:
            composer = file['metadata']['composer']
            genre = file['metadata']['genre']
            # 统计作曲家分布
            if composer in stats['composer_distribution']:
                stats['composer_distribution'][composer] += 1
            else:
                stats['composer_distribution'][composer] = 1
            # 统计体裁分布
            if genre in stats['genre_distribution']:
                stats['genre_distribution'][genre] += 1
            else:
                stats['genre_distribution'][genre] = 1
        # 保存统计信息
        stats_path = os.path.join(output_dir, 'processing_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
