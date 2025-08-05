import streamlit as st
import os
import json

color_same = 'grey'
color_different = 'black'

def app():
    st.title("Recap")
    
    experiment_dir = "experiments"

    experiments = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]

    selected_experiments = st.multiselect("Select up to two experiments to review",
                                          options = experiments,
                                          max_selections=2
                                          )
    
    def parameter_review(exp) :
        parameter_path = os.path.join(experiment_dir, exp, "parameters.json")
        with open(parameter_path, "r") as f:
            return json.load(f)
        return None
    
    if len(selected_experiments) == 2:
        exp1, exp2 = selected_experiments
        col1, col2 = st.columns(2)
        
        exp1_param = parameter_review(exp1)
        exp2_param = parameter_review(exp2)

        with col1 :

            st.markdown("<b>Experiment Name</b>", unsafe_allow_html=True)
            st.markdown(exp1_param['experiment_name'])

            st.markdown("<b>Data Source</b>", unsafe_allow_html=True)
            color = color_same if exp1_param["data_source"] == exp2_param["data_source"] else color_different
            st.markdown(
                f"<span style='color:{color}'>"
                + f"Dataset: {exp1_param['data_source']['dataset']}"
                + "<br/>"
                + f"Preprocessing: {exp1_param['data_source']['preprocessing']}"
                + "</span>",
                unsafe_allow_html=True
            )

            st.markdown("<b>Model</b>", unsafe_allow_html=True)
            color = color_same if exp1_param["model"] == exp2_param["model"] else color_different
            string = (f"<span style='color:{color}'>"
                      + f"Type: {exp1_param['model']['type']}"
                      + "</span>"
                      + f"<ul style='color:{color}'>")
            for key, item in exp1_param['model']['hyperparameters'].items() :
                string += f"<li>{key}: {item}</li>"
            string += "</ul>"
            st.markdown(string, unsafe_allow_html=True)

            st.markdown("<b>Training</b>", unsafe_allow_html=True)
            color = color_same if exp1_param["split_param"] == exp2_param["split_param"] else color_different
            st.markdown(
                f"<span style='color:{color}'>"
                + f"Target feature: {exp1_param['split_param']['target']}"
                + "</br>"
                + f"Protected feature: {exp1_param['split_param']['protected']}"
                + "</br>"
                + f"Test size: {exp1_param['split_param']['test_size']}"
                + "</br>"
                + f"Train Random seed: {exp1_param['split_param']['seed']}",
                unsafe_allow_html=True
            )

            st.markdown("<b>Performance</b>", unsafe_allow_html=True)
            string = ""
            for metric in exp1_param['performance_metrics'] :
                string += f"{metric}: {round(exp1_param['performance_measurments'][metric], 3)}<br/>"
            st.markdown(string, unsafe_allow_html=True)

            st.markdown("<b>Fairness</b>", unsafe_allow_html=True)
                    
            string = ""
            for metric in exp1_param['fairness_metrics'] :
                string += f"{metric}: {round(exp1_param['fairness_measurments'][metric], 3)}<br/>"
            st.markdown(string, unsafe_allow_html=True)

            if len(exp1_param["fairness_treatment"]) > 0 :
                string = "Fairness Treatments:<ul>"
                for t in exp1_param["fairness_treatment"] :
                    string += f"<li>{t}"
                    if exp1_param["fairness_treatment_param"][t] != {} :
                        string += " ("
                        for key, item in exp1_param["fairness_treatment_param"][t].items() :
                            string += f"{key}: {item}, "
                        string = string[:-2]
                        string += ")"
                    string += "</li>"
                string += "</ul>"
                st.markdown(string, unsafe_allow_html=True)


        with col2 : 

            st.markdown("<b>Experiment Name</b>", unsafe_allow_html=True)
            st.markdown(exp2_param['experiment_name'])

            st.markdown("<b>Data Source</b>", unsafe_allow_html=True)
            color = color_same if exp1_param["data_source"] == exp2_param["data_source"] else color_different
            st.markdown(
                f"<span style='color:{color}'>"
                + f"Dataset: {exp2_param['data_source']['dataset']}"
                + "<br/>"
                + f"Preprocessing: {exp2_param['data_source']['preprocessing']}"
                + "</span>",
                unsafe_allow_html=True
            )

            st.markdown("<b>Model</b>", unsafe_allow_html=True)
            color = color_same if exp1_param["model"] == exp2_param["model"] else color_different
            string = (f"<span style='color:{color}'>"
                + f"Type: {exp2_param['model']['type']}"
                + "</span>"
                + f"<ul style='color:{color}'>")
            for key, item in exp2_param['model']['hyperparameters'].items() :
                string += f"<li>{key}: {item}</li>"
            string += "</ul>"
            st.markdown(string, unsafe_allow_html=True)

            st.markdown("<b>Training</b>", unsafe_allow_html=True)
            color = color_same if exp1_param["split_param"] == exp2_param["split_param"] else color_different
            st.markdown(
                f"<span style='color:{color}'>"
                + f"Target feature: {exp2_param['split_param']['target']}"
                + "</br>"
                + f"Protected feature: {exp2_param['split_param']['protected']}"
                + "</br>"
                + f"Test size: {exp2_param['split_param']['test_size']}"
                + "</br>"
                + f"Train Random seed: {exp2_param['split_param']['seed']}",
                unsafe_allow_html=True
            )

            st.markdown("<b>Performance</b>", unsafe_allow_html=True)
            string = ""
            for metric in exp2_param['performance_metrics'] :
                string += f"{metric}: {round(exp2_param['performance_measurments'][metric], 3)}<br/>"
            st.markdown(string, unsafe_allow_html=True)

            st.markdown("<b>Fairness</b>", unsafe_allow_html=True)
                    
            string = ""
            for metric in exp2_param['fairness_metrics'] :
                string += f"{metric}: {round(exp2_param['fairness_measurments'][metric], 3)}<br/>"
            st.markdown(string, unsafe_allow_html=True)

            if len(exp2_param["fairness_treatment"]) > 0 :
                string = "Fairness Treatments:<ul>"
                for t in exp2_param["fairness_treatment"] :
                    string += f"<li>{t}"
                    if exp2_param["fairness_treatment_param"][t] != {} :
                        string += " ("
                        for key, item in exp2_param["fairness_treatment_param"][t].items() :
                            string += f"{key}: {item}, "
                        string = string[:-2]
                        string += ")"
                    string += "</li>"
                string += "</ul>"
                st.markdown(string, unsafe_allow_html=True)

    elif len(selected_experiments) == 1:

        exp1 = selected_experiments[0]
        exp1_param = parameter_review(exp1)

        st.markdown("<b>Experiment Name</b>", unsafe_allow_html=True)
        st.markdown(exp1_param['experiment_name'])

        st.markdown("<b>Data Source</b>", unsafe_allow_html=True)
        st.markdown(
            f"Dataset: {exp1_param['data_source']['dataset']}"
            + "<br/>"
            + f"Preprocessing: {exp1_param['data_source']['preprocessing']}",
            unsafe_allow_html=True
        )

        st.markdown("<b>Model</b>", unsafe_allow_html=True)
        string = (f"Type: {exp1_param['model']['type']}"
                +"<ul>")
        for key, item in exp1_param['model']['hyperparameters'].items() :
            string += f"<li>{key}: {item}</li>"
        string += "</ul>"
        st.markdown(string, unsafe_allow_html=True)

        st.markdown("<b>Training</b>", unsafe_allow_html=True)
        st.markdown(
            f"Target feature: {exp1_param['split_param']['target']}"
            + "</br>"
            + f"Protected feature: {exp1_param['split_param']['protected']}"
            + "</br>"
            + f"Test size: {exp1_param['split_param']['test_size']}"
            + "</br>"
            + f"Train Random seed: {exp1_param['split_param']['seed']}",
            unsafe_allow_html=True
        )

        st.markdown("<b>Performance</b>", unsafe_allow_html=True)
        string = ""
        for metric in exp1_param['performance_metrics'] :
            string += f"{metric}: {round(exp1_param['performance_measurments'][metric], 3)}<br/>"
        st.markdown(string, unsafe_allow_html=True)

        st.markdown("<b>Fairness</b>", unsafe_allow_html=True)
                
        string = ""
        for metric in exp1_param['fairness_metrics'] :
            string += f"{metric}: {round(exp1_param['fairness_measurments'][metric], 3)}<br/>"
        st.markdown(string, unsafe_allow_html=True)

        if len(exp1_param["fairness_treatment"]) > 0 :
            string = "Fairness Treatments:<ul>"
            for t in exp1_param["fairness_treatment"] :
                string += f"<li>{t}"
                if exp1_param["fairness_treatment_param"][t] != {} :
                    string += " ("
                    for key, item in exp1_param["fairness_treatment_param"][t].items() :
                        string += f"{key}: {item}, "
                    string = string[:-2]
                    string += ")"
                string += "</li>"
            string += "</ul>"
            st.markdown(string, unsafe_allow_html=True)