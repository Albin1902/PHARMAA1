# ==================== Patient Profile Analyzer Tab ====================
with tabs[1]:
    st.header("Patient Profile Analyzer")
    st.caption("Upload a screenshot of the patient's Healthwatch profile. Specify what medication type you're looking for (e.g., 'diabetes', 'cholesterol', 'blood pressure').")

    prompt = st.text_input(
        "What medication are you looking for?",
        placeholder="e.g., diabetes medication, statin for cholesterol, blue inhaler, blood thinner",
        key="analyzer_prompt"
    )

    uploaded_file = st.file_uploader(
        "Upload screenshot (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="analyzer_upload"
    )

    if uploaded_file and prompt:
        try:
            from PIL import Image, ImageEnhance
            import pytesseract
        except ImportError as e:
            st.error("Missing required packages. Add to requirements.txt:\n\npillow\npytesseract")
            st.stop()

        with st.spinner("Extracting text from screenshot..."):
            img = Image.open(uploaded_file)
            # Preprocessing for better OCR accuracy
            img = img.convert('L')  # grayscale
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(img, lang='eng+fra').lower()

        with st.expander("Raw extracted text (for debugging)", expanded=False):
            st.text(text)

        # Detect medications in extracted text
        found_meds = []
        for _, row in df.iterrows():
            brand = normalize_text(row.get('brand_name', ''))
            generic = normalize_text(row.get('generic_name', ''))
            synonyms = normalize_text(row.get('synonyms', ''))
            if (brand and brand in text) or \
               (generic and generic in text) or \
               (synonyms and any(normalize_text(w) in text for w in synonyms.split() if len(w) > 3)):
                found_meds.append(row)

        if not found_meds:
            st.info("No known medications from the database were detected in the screenshot.")
            st.stop()

        found_df = pd.DataFrame(found_meds)

        # Filter by user prompt
        expanded_prompt = expand_query(prompt)
        if expanded_prompt.strip():
            blobs = found_df.apply(row_search_blob, axis=1)
            mask = blobs.apply(lambda b: matches_query(b, expanded_prompt))
            matching_df = found_df[mask]
        else:
            matching_df = found_df

        if matching_df.empty:
            st.warning(f"Medications detected, but none match your request ('{prompt}'). Showing all detected:")
            st.dataframe(found_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.success("✅ Matching medication(s) found for refill!")
            st.write("The following appear in the patient's profile and match your query:")

            for _, med in matching_df.iterrows():
                brand = med.get('brand_name', 'Unknown')
                st.markdown(f"### 💊 **{brand}** – likely needs refill")

                details = f"""
- **Generic name:** {med.get('generic_name', '')}
- **Category:** {med.get('category', '')}
- **Form:** {med.get('form', '')}
- **DIN:** {med.get('DIN', '')}
- **Synonyms:** {med.get('synonyms', '')}
"""
                st.markdown(details)

                # === ADD IMAGES HERE ===
                imgs = find_images_for_brand_name(brand)

                if not (imgs["box"] or imgs["pill"] or imgs["other"]):
                    st.caption("ℹ️ No local images available for this medication yet.")
                else:
                    box_img = pick_first(imgs["box"])
                    pill_img = pick_first(imgs["pill"])

                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Box**")
                        if box_img:
                            st.image(str(box_img), use_container_width=True)
                        else:
                            st.caption("No box image")

                    with cols[1]:
                        st.write("**Pill / Inhaler**")
                        if pill_img:
                            st.image(str(pill_img), use_container_width=True)
                        else:
                            st.caption("No pill/inhaler image")

                    # Optional: show extra images
                    remaining = imgs["box"][1:] + imgs["pill"][1:] + imgs["other"]
                    if remaining:
                        with st.expander("More images"):
                            st.image([str(p) for p in remaining], use_container_width=True)

                st.divider()  # nice separator between multiple meds

            # Final table of all matching results
            st.dataframe(matching_df[display_cols], use_container_width=True, hide_index=True)
