#!/usr/bin/env python3
"""
BC AI Survey Data Validation Script
==================================

Quick validation and preview of the BC AI Survey dataset before processing.
"""

from pathlib import Path

import pandas as pd


def validate_data():
    """Validate the BC AI Survey data and show preview"""

    # Find the data file
    data_file = "../Hackathon round 3 with demos[48].csv"
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print(
            "Make sure you're running this script from the bc_ai_hackathon_round_3 directory."
        )
        return False

    print("✅ BC AI Survey Data Validation")
    print("=" * 50)

    try:
        # Load the data
        print(f"📊 Loading survey data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} responses with {len(df.columns)} columns")
        print()

        # Check the text columns we want to process
        target_columns = [
            "Q2_Experience_with_AI_animal_other_OE",
            "Q4A_Sector_AI_making_a_positive_impact_other_OE",
            "Q4B_Sector_AI_making_a_negative_impact_other_OE",
            "Q5_Art_vibe_AI_other_OE",
            "Q8_AI_helping_BC_community_text_OE",
            "Q8_AI_helping_BC_community_video_OE_transcription",
            "Q10_AI_superpower_to_help_community_other_OE",
            "Q11_Gainer_advances_in_AI_other_OE",
            "Q12_Most_trustworthy_advances_in_AI_other_OE",
            "Q13_AI_impact_worries_text_OE",
            "Q13_AI_impact_worries_videos_OE_transcription",
            "Q15_Matters_most_BC_AI_future_other_OE",
            "Q16_Indigenous_communities_involvement_AI_text_OE",
            "Q16_Indigenous_communities_involvement_AI_video_OE_transcription",
            "Q17_Advice_BC_Leaders_text_OE",
            "Q17_Advice_BC_Leaders_video_OE_transcription",
        ]

        print("📝 Text Column Validation:")
        valid_columns = []
        for col in target_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                non_empty_count = df[col].fillna("").str.strip().str.len().gt(0).sum()
                print(f"   ✅ {col}: {non_empty_count} valid responses")
                if non_empty_count > 0:
                    valid_columns.append(col)
            else:
                print(f"   ❌ {col}: Column not found")

        print(f"\n📊 Summary:")
        print(f"   • Target columns: {len(target_columns)}")
        print(f"   • Valid columns: {len(valid_columns)}")
        print(f"   • Missing columns: {len(target_columns) - len(valid_columns)}")

        if len(valid_columns) > 0:
            print(f"\n🎯 Ready to process {len(valid_columns)} text columns!")

            # Show sample responses
            print(f"\n📖 Sample Responses:")
            for col in valid_columns[:3]:  # Show first 3 columns
                sample_responses = df[col].dropna().head(2)
                print(f"\n   {col}:")
                for i, response in enumerate(sample_responses, 1):
                    preview = (
                        response[:100] + "..." if len(response) > 100 else response
                    )
                    print(f'      {i}. "{preview}"')

            print(f"\n🚀 Next Steps:")
            print(f"   1. Set up your OpenAI API key in .env file")
            print(f"   2. Run: python embeddings_of_data.py")
            print(f"   3. Run: python visualizations_3d_silhouette_focus_sized.py")
            print(f"   4. Open: index.html")

            return True
        else:
            print(f"\n❌ No valid text columns found. Check the dataset structure.")
            return False

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


if __name__ == "__main__":
    success = validate_data()
    if success:
        print(f"\n✅ Validation complete - ready to proceed!")
    else:
        print(f"\n❌ Validation failed - please check the data file.")
