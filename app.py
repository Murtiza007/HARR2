from flask import Flask,render_template,request
import pickle
import numpy as np
import joblib
app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/harr')
def harr():
    return render_template('harr.html')

@app.route('/riasec')
def riasec():
    return render_template('riasec.html')

@app.route('/mbti')
def mbti():
    return render_template('mbti.html')

@app.route('/gardener')
def gardener():
    return render_template('gardener.html')

@app.route('/instructionsWERP',methods=["GET","POST"])
def instructionsWERP():
    if (request.method=="GET"):
        return render_template('instructionsWERP.html')
    else: 
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        phone = request.form.get('phone')
        grade = request.form.get('grade')
        

        # Pass data to HolisticAcademic.html
        return render_template('HolisticAcademicWERP.html',name=name,gender=gender,age=age,email=email,phone=phone,grade=grade)






@app.route('/instructions',methods=["GET","POST"])
def instructions():
    if (request.method=="GET"):
        return render_template('instructions.html')
    else: 
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        phone = request.form.get('phone')
        grade = request.form.get('grade')
        

        # Pass data to HolisticAcademic.html
        return render_template('HolisticAcademic.html',name=name,gender=gender,age=age,email=email,phone=phone,grade=grade)
   

@app.route('/HolisticAcademic', methods=["GET", "POST"])
def HolisticAcademic():
    if request.method == "POST":
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        phone = request.form.get('phone')
        grade = request.form.get('grade')
        return render_template('HolisticAcademic.html', name=name,gender=gender,age=age,email=email,phone=phone,grade=grade)
    else:
        return render_template('HolisticAcademic.html')



@app.route('/HolisticAcademicWERP', methods=["GET", "POST"])
def HolisticAcademicWERP():
    if request.method == "POST":
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        phone = request.form.get('phone')
        grade = request.form.get('grade')
        return render_template('HolisticAcademicWERP.html', name=name,gender=gender,age=age,email=email,phone=phone,grade=grade)
    else:
        return render_template('HolisticAcademicWERP.html')


@app.route('/submitAcademic', methods=["GET", "POST"])
def submitAcademic():
    if request.method == "POST":
      
        math = float(request.form.get('math_score'))
        science = float(request.form.get('science_score'))
        language = float(request.form.get('language_score'))
        social_score = float(request.form.get('social_score'))
        cs = float(request.form.get('cs_score'))
        pe = float(request.form.get('pe_score'))
        percentage = float(request.form.get('percentage'))
        
        logic = float(request.form.get('logic'))
        verbal = float(request.form.get('verbal'))
        creativity = float(request.form.get('creative'))
        analytical = float(request.form.get('analytical'))
        leadership = float(request.form.get('leadership'))
        problem = request.form.get('problem')
        decision = request.form.get('decision')
        social = request.form.get('social')
        subject= request.form.get('subject')
        
        hobby = request.form.get('hobby')
        sports = request.form.get('sports')
        
        med_score=round((0.1*math)+(0.35*science)+(0.1*language)+(0.05*social_score)+(0.1*cs)+(0.1*logic)+(0.05*creativity)+(0.05*analytical)+(0.05*verbal)+(0.05*leadership),3)
        non_med_score=round((0.25*math)+(0.20*science)+(0.05*language)+(0.05*social_score)+(0.15*cs)+(0.1*logic)+(0.05*creativity)+(0.05*analytical)+(0.05*verbal)+(0.05*leadership),3)
        commerce_score=round((0.2*math)+(0.05*science)+(0.1*language)+(0.15*social_score)+(0.1*cs)+(0.1*logic)+(0.05*creativity)+(0.15*analytical)+(0.05*verbal)+(0.05*leadership),3)
        arts_score=round((0.05*math)+(0.05*science)+(0.15*language)+(0.25*social_score)+(0.05*cs)+(0.05*logic)+(0.15*creativity)+(0.05*analytical)+(0.15*verbal)+(0.05*leadership),3)
        
        best_score= {
            "Medical": med_score,
            "Non-Medical": non_med_score,
            "Commerce": commerce_score,
            "Arts": arts_score
        }

        best_stream = max(best_score, key=best_score.get)  # this gives you the name
        sorted_streams = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
        second_best_stream = sorted_streams[1][0]

        
        return render_template('submitAcademic.html',math=math,science=science,
    language=language,
    social_score=social_score,
    cs=cs,
    pe=pe,
    percentage=percentage,
    logic=logic,
    verbal=verbal,
    creativity=creativity,
    analytical=analytical,
    leadership=leadership,
    problem=problem,
    decision=decision,
    social_skills=social,
    subject=subject,hobby=hobby,sports=sports,med_score=med_score,non_med_score=non_med_score,commerce_score=commerce_score,arts_score=arts_score,best_stream=best_stream,second_best_stream=second_best_stream)
   
        
@app.route('/improvementindexWERP', methods=["GET", "POST"])
def improvementindexWERP():
   if request.method == "POST":
      
        math = float(request.form.get('math_score'))
        science = float(request.form.get('science_score'))
        language = float(request.form.get('language_score'))
        social_score = float(request.form.get('social_score'))
        cs = float(request.form.get('cs_score'))
        pe = float(request.form.get('pe_score'))
        percentage = float(request.form.get('percentage'))
        
        logic = float(request.form.get('logic'))
        verbal = float(request.form.get('verbal'))
        creativity = float(request.form.get('creative'))
        analytical = float(request.form.get('analytical'))
        leadership = float(request.form.get('leadership'))
        problem = request.form.get('problem')
        decision = request.form.get('decision')
        social = request.form.get('social')
        subject= request.form.get('subject')
        
        hobby = request.form.get('hobby')
        sports = request.form.get('sports')
        
        med_score=round((0.1*math)+(0.35*science)+(0.1*language)+(0.05*social_score)+(0.1*cs)+(0.1*logic)+(0.05*creativity)+(0.05*analytical)+(0.05*verbal)+(0.05*leadership),3)
        non_med_score=round((0.25*math)+(0.20*science)+(0.05*language)+(0.05*social_score)+(0.15*cs)+(0.1*logic)+(0.05*creativity)+(0.05*analytical)+(0.05*verbal)+(0.05*leadership),3)
        commerce_score=round((0.2*math)+(0.05*science)+(0.1*language)+(0.15*social_score)+(0.1*cs)+(0.1*logic)+(0.05*creativity)+(0.15*analytical)+(0.05*verbal)+(0.05*leadership),3)
        arts_score=round((0.05*math)+(0.05*science)+(0.15*language)+(0.25*social_score)+(0.05*cs)+(0.05*logic)+(0.15*creativity)+(0.05*analytical)+(0.15*verbal)+(0.05*leadership),3)
        
        best_score= {
            "Medical": med_score,
            "Non-Medical": non_med_score,
            "Commerce": commerce_score,
            "Arts": arts_score
        }

        best_stream = max(best_score, key=best_score.get)  # this gives you the name
        sorted_streams = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
        second_best_stream = sorted_streams[1][0]

        y_math = float(request.form.get('y_math_score'))
        y_science = float(request.form.get('y_science_score'))
        y_language = float(request.form.get('y_language_score'))
        y_social_score = float(request.form.get('y_social_score'))
        y_cs = float(request.form.get('y_cs_score'))
        y_pe = float(request.form.get('y_pe_score'))
        y_percentage = float(request.form.get('y_percentage'))
        
        z_math = float(request.form.get('z_math_score'))
        z_science = float(request.form.get('z_science_score'))
        z_language = float(request.form.get('z_language_score'))
        z_social_score = float(request.form.get('z_social_score'))
        z_cs = float(request.form.get('z_cs_score'))
        z_pe = float(request.form.get('z_pe_score'))
        z_percentage = float(request.form.get('z_percentage'))
        
        
        return render_template('improvementindexWERP.html',math=math,science=science,
    language=language,
    social_score=social_score,
    cs=cs,
    pe=pe,
    percentage=percentage,
    logic=logic,
    verbal=verbal,
    creativity=creativity,
    analytical=analytical,
    leadership=leadership,
    problem=problem,
    decision=decision,
    social_skills=social,
    subject=subject,hobby=hobby,sports=sports,med_score=med_score,non_med_score=non_med_score,commerce_score=commerce_score,arts_score=arts_score,best_stream=best_stream,second_best_stream=second_best_stream,)
   
   
@app.route('/riasecWERP', methods=["GET", "POST"])
def riasecWERP():
    if request.method == "POST":
        
       
        
        
        y_math = float(request.form.get('y_math_score'))
        y_science = float(request.form.get('y_science_score'))
        y_language = float(request.form.get('y_language_score'))
        y_social_score = float(request.form.get('y_social_score'))
        y_cs = float(request.form.get('y_cs_score'))
        y_pe = float(request.form.get('y_pe_score'))
        y_percentage = float(request.form.get('y_percentage'))
        
        z_math = float(request.form.get('z_math_score'))
        z_science = float(request.form.get('z_science_score'))
        z_language = float(request.form.get('z_language_score'))
        z_social_score = float(request.form.get('z_social_score'))
        z_cs = float(request.form.get('z_cs_score'))
        z_pe = float(request.form.get('z_pe_score'))
        z_percentage = float(request.form.get('z_percentage'))
        
        
     
        return render_template(
    'riasecWERP.html',
    y_math=y_math,
    y_science=y_science,
    y_language=y_language,
    y_social_score=y_social_score,
    y_cs=y_cs,
    y_pe=y_pe,
    y_percentage=y_percentage,
    z_math=z_math,
    z_science=z_science,
    z_language=z_language,
    z_social_score=z_social_score,
    z_cs=z_cs,
    z_pe=z_pe,
    z_percentage=z_percentage
)
    else:
        return render_template('riasecWERP.html')




@app.route('/predict',methods=["GET","POST"])
def predict():
    if (request.method=="GET"):
        return render_template('home.html')
    else:   
       model = joblib.load('model (1).pkl')
       le = joblib.load('label_encoder (1).pkl')
       
        
    math = float(request.form.get('math_score'))
    science = float(request.form.get('science_score'))
    language = float(request.form.get('language_score'))
    social_score = float(request.form.get('social_score'))
    cs = float(request.form.get('cs_score'))
    pe = float(request.form.get('pe_score'))
    percentage = float(request.form.get('percentage'))
        
    logic = float(request.form.get('logic'))
    verbal = float(request.form.get('verbal'))
    creativity = float(request.form.get('creative'))
    analytical = float(request.form.get('analytical'))
    leadership = float(request.form.get('leadership'))
    problem = request.form.get('problem')
    decision = request.form.get('decision')
    social = request.form.get('social')
    subject= request.form.get('subject')
        
    med_score=round((0.1*math)+(0.35*science)+(0.1*language)+(0.05*social_score)+(0.1*cs)+(0.1*logic)+(0.05*creativity)+(0.05*analytical)+(0.05*verbal)+(0.05*leadership),3)
    non_med_score=round((0.25*math)+(0.20*science)+(0.05*language)+(0.05*social_score)+(0.15*cs)+(0.1*logic)+(0.05*creativity)+(0.05*analytical)+(0.05*verbal)+(0.05*leadership),3)
    commerce_score=round((0.2*math)+(0.05*science)+(0.1*language)+(0.15*social_score)+(0.1*cs)+(0.1*logic)+(0.05*creativity)+(0.15*analytical)+(0.05*verbal)+(0.05*leadership),3)
    arts_score=round((0.05*math)+(0.05*science)+(0.15*language)+(0.25*social_score)+(0.05*cs)+(0.05*logic)+(0.15*creativity)+(0.05*analytical)+(0.15*verbal)+(0.05*leadership),3)
        
        
    logic = float(request.form.get('logic'))
    verbal = float(request.form.get('verbal'))
    creativity = float(request.form.get('creative'))
    analytical = float(request.form.get('analytical'))
    leadership = float(request.form.get('leadership'))
    problem = request.form.get('problem')
    decision = request.form.get('decision')
    social = request.form.get('social')
    subject= request.form.get('subject')
    hobby = request.form.get('hobby')
    sports = request.form.get('sports')
        
       
        
    y_math = float(request.form.get('y_math_score'))
    y_science = float(request.form.get('y_science_score'))
    y_language = float(request.form.get('y_language_score'))
    y_social_score = float(request.form.get('y_social_score'))
    y_cs = float(request.form.get('y_cs_score'))
    y_pe = float(request.form.get('y_pe_score'))
    y_percentage = float(request.form.get('y_percentage'))
        
    z_math = float(request.form.get('z_math_score'))
    z_science = float(request.form.get('z_science_score'))
    z_language = float(request.form.get('z_language_score'))
    z_social_score = float(request.form.get('z_social_score'))
    z_cs = float(request.form.get('z_cs_score'))
    z_pe = float(request.form.get('z_pe_score'))
    z_percentage = float(request.form.get('z_percentage'))
    
    riasec_r = request.form.get("riasec_r")
    riasec_i = request.form.get("riasec_i")
    riasec_a = request.form.get("riasec_a")
    riasec_s = request.form.get("riasec_s")
    riasec_e = request.form.get("riasec_e")
    riasec_c = request.form.get("riasec_c")
    riasec_code = request.form.get("riasec_code")
        
  

    features = [
    float(med_score),
    float(non_med_score),
    float(commerce_score),
    float(arts_score),
 
            # Same here – may need encoding
]

# Convert list to numpy 2D array
    final_features = np.array([features])
    prediction = model.predict(final_features)
    prediction_label = le.inverse_transform(prediction)[0]
                
        
    return render_template('predict.html',math=math,science=science,
    language=language,
    social_score=social_score,
    cs=cs,
    pe=pe,
    percentage=percentage,
    logic=logic,
    verbal=verbal,
    creativity=creativity,
    analytical=analytical,
    leadership=leadership,
    problem=problem,
    decision=decision,
    social_skills=social,
    subject=subject,hobby=hobby,sports=sports,y_math=y_math,
    y_science=y_science,
    y_language=y_language,
    y_social_score=y_social_score,
    y_cs=y_cs,
    y_pe=y_pe,
    y_percentage=y_percentage,
    z_math=z_math,
    z_science=z_science,
    z_language=z_language,
    z_social_score=z_social_score,
    z_cs=z_cs,
    z_pe=z_pe,
    z_percentage=z_percentage,riasec_code=riasec_code,med_score=med_score,non_med_score=non_med_score,commerce_score=commerce_score,arts_score=arts_score,prediction_label=prediction_label)
   
   
   
           
    
    #final_features=[np.array(int_features)]
    
    #prediction = model.predict(final_features)
    #prediction_label = le.inverse_transform(prediction)[0]
    
    
    return render_template ('predict.html')
       
       
        
    

if __name__=='__main__':
    app.run(debug=True)

