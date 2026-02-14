#!/usr/bin/env python3
"""
اجرای کامل پروژه در Colab
"""

import os
import sys
from datetime import datetime


def print_menu():
    print("\n" + "="*70)
    print("🎓 پروژه مدلسازی داده‌های کنکور ایران")
    print("="*70)
    print("1️⃣  مرحله ۱: مدل‌های پایه (MLP, RF, GBM)")
    print("2️⃣  مرحله ۲: TabTransformer")
    print("3️⃣  مرحله ۳: جاسازی عددی")
    print("4️⃣  مرحله ۴: تحلیل نهایی")
    print("5️⃣  همه مراحل")
    print("0️⃣  خروج")
    print("="*70)


def run_stage1():
    print("\n🚀 اجرای مرحله ۱...")
    import stage1_baseline
    return stage1_baseline.run_stage1()


def run_stage2():
    print("\n🚀 اجرای مرحله ۲...")
    import stage2_tabtransformer
    return stage2_tabtransformer.run_stage2()


def run_stage3():
    print("\n🚀 اجرای مرحله ۳...")
    import stage3_numerical_embeddings
    return stage3_numerical_embeddings.run_stage3()


def run_stage4():
    print("\n🚀 اجرای مرحله ۴...")
    import stage4_final_analysis
    return stage4_final_analysis.run_stage4()


def run_all():
    print("\n🚀 اجرای همه مراحل...")
    run_stage1()
    run_stage2()
    run_stage3()
    run_stage4()
    print("\n✅ همه مراحل با موفقیت اجرا شدند!")


def main():
    while True:
        print_menu()
        choice = input("👉 انتخاب شما: ").strip()
        
        if choice == '1':
            run_stage1()
        elif choice == '2':
            run_stage2()
        elif choice == '3':
            run_stage3()
        elif choice == '4':
            run_stage4()
        elif choice == '5':
            run_all()
        elif choice == '0':
            print("\n👋 خداحافظ!")
            break
        else:
            print("\n❌ انتخاب نامعتبر!")
        
        input("\n⏎ Enter را بزنید...")


if __name__ == "__main__":
    main()
