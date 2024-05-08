def model_namer(model_name):
  if model_name == 'ResNet50':
        model_name = 'r50'
  elif 'Robust' in model_name:
      # get model name after 'robust'
      main_name = model_name.split('Robust')[1].lower()[:3]
      model_name = f'AT_{model_namer(main_name)}'
  else:
      model_name = model_name.lower()[:3]
  return model_name