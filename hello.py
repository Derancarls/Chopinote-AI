from music21 import *
import music21

print("=" * 50)
print("music21 测试")
print("=" * 50)

# 修复：使用 music21.__version__
print(f"\n1. music21 版本: {music21.__version__}")

# 测试2：解析内置语料库中的巴赫作品
print("\n2. 测试解析巴赫作品...")
bach_chorale = corpus.parse('bach/bwv66.6')
print(f"   ✓ 成功解析")
print(f"   作品标题: {bach_chorale.metadata.title}")
print(f"   作曲家: {bach_chorale.metadata.composer}")
print(f"   声部数: {len(bach_chorale.parts)}")

# 测试3：查看乐谱基本信息
print("\n3. 乐谱基本信息:")
print(f"   总音符数: {len(bach_chorale.flat.notes)}")
print(f"   拍号: {bach_chorale.timeSignature}")
print(f"   调号: {bach_chorale.keySignature}")

# 测试4：显示前10个音符
print("\n4. 前10个音符:")
for i, note in enumerate(bach_chorale.flat.notes[:10]):
    print(f"   {i+1}. {note.name}{note.octave} ({note.quarterLength}拍)")

print("\n" + "=" * 50)
print("所有测试通过！music21 工作正常！")
print("=" * 50)

