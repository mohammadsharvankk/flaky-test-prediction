"""
Final step: embeds dashboard_data.json, predictor_ui.json, predictor_trees.json
into dashboard_template.html to produce the finished, self-contained dashboard.

Run this LAST, after:
  1. prepare_dashboard_data.py   -> dashboard_data.json
  2. export_predictor_model.py   -> predictor_ui.json, predictor_trees.json
"""

with open('dashboard/dashboard_template.html') as f:
    template = f.read()
with open('dashboard/build/dashboard_data.json') as f:
    data = f.read()
with open('dashboard/build/predictor_ui.json') as f:
    ui = f.read()
with open('dashboard/build/predictor_trees.json') as f:
    trees = f.read()

final = template.replace('__DATA_PLACEHOLDER__', data)
final = final.replace('__PREDICTOR_UI_PLACEHOLDER__', ui)
final = final.replace('__PREDICTOR_TREES_PLACEHOLDER__', trees)

assert '__DATA_PLACEHOLDER__' not in final
assert '__PREDICTOR_UI_PLACEHOLDER__' not in final
assert '__PREDICTOR_TREES_PLACEHOLDER__' not in final

with open('dashboard/flakeguard_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(final)

print(f"Saved flakeguard_dashboard.html - {len(final) / 1024 / 1024:.2f} MB")
print("Open it in your browser to check everything looks right.")
