for zip in *.zip; do unzip -j "$zip" '*.csv'; done
rm *.zip
