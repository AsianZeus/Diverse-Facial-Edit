# Diverse-Facial-Edit
<p>Diverse Facial Edit with StyleGAN, StyleGAN2, StyleClip with ViT, and Other Features like Background Removal and Face Swap.
This API serve the purpose of generating random real looking images of human faces which can be edited in latent space and can generate new images from the given parameters with extra feauture of removing background and faceswap.</p>


<hr><hr>
## API Documentation:
Flow:
1) Login [Generate a Session ID]
2) Generate random faces of male female or randomly
3) Global Edit (i.e glasses, facialhair, age, gender, smile, headPose [left right])
4) Pitch Edit [Headpose Up Down]
5) Local Edit (i.e Ethenicity, Hair Color, Hair Highlights, Hairstyle, Facial Hair, Accesories, Emotions, Eye Color, Eye Size, Face Shape, Makeup, Ear Size, Lip Size, Facespots)
6) Background Removal
7) Face Swap
<hr><hr>

## API Calls
<hr>
1. Login:
    - Description: Generate a session id for you for 15 mins of usage
    - Url: /login
    - Type: GET
    - Response: JSON
    - Example-Response:

            {
                "id": "fec42e0311f7e1f0"
            }

<hr>
2. Generate Random Face
    - Description: Generate a random image for you based on the given parameters
    - Url: /generateRandomFace
    - Type: POST
    - Body-Type: JSON 
    - Body: 

            {
                "id":"fec42e0311f7e1f0",
                "filter":"r"
            }

    - Parameters-Description:
        * id: Session ID generated via /login
        * filter: takes a string 
            -    "m" for male
            -    "f" for female
            -    "r" for random
    - Response: JSON
    - Example-Response:

            {
                "status": "Success"
            }

<hr>
3. Gloal Edit:
    - Description: Allows you to gloablly edit your generated image based on parametric values given, The global edit will take the parameters in bulk and then edit your image with multiple facial edits parallely,
    - NOTE: In case you'll do this api call again, it will edit your image from scratch (i.e generated image and not the recently edited image) so you can play with different permutation combination of gobal edits in one go.
        -   You cannot call globalEdit after using pitchEdit or LocalEdit since GlobalEdit uses a different latent space, it cannot be encoded again back, so if you want to call globalEdit you can call it in the beginning or before calling pitchEdit or localEdit.
    - Url: /globalEdit
    - Type: POST
    - Body-Type: JSON 
    - Body:

            {
                "id": "fec42e0311f7e1f0",
                "attr_values": {
                    "glasses_direction": 2,
                    "facialhair_direction": 0,
                    "age_w": 0,
                    "gender_w": 0,
                    "smile_w": 0,
                    "headPose_roll_direction": 2
                }
            }

    - Parameters-Description:
        * id: Session ID generated via /login
        * attr_values: 
            + key: the facial feature to edit-> [glasses_direction","facialhair_direction","age_w","gender_w","smile_w","headPose_roll_direction"]
            + values: the alpha value ranges from -3 to 3, implies how strongly do you want to apply that edit for example "simle_w" at 0 will give a neutral face and "smile_w" at 2 or 3 will exibit a face with strong smile.
    - Response: JSON
    - Example-Response:

            {
                "status": "Success"
            }

<hr>
4) Pitch Edit:
Description: Allows you to edit your face's headpose pitch, i.e move in up or down direction.
NOTE: You cannot do any globalEdits after calling pitchEdit.
Url: /pitchEdit
Type: POST
Body-Type: JSON 
Body:
    {
        "id":"fec42e0311f7e1f0",
        "latent_weight":7
        "on_original":0
    }

Parameters-Description:
    a) id: Session ID generated via /login
    b) latent_weight: the alpha value ranges from -10 to 10, implies how strongly do you want to apply that edit for example "-10" will give a face looking in up direction and "10" will exibit a face facing in down direction.
    c) on_original: takes a boolean value i.e 0 or 1, to decide weather to do the operation on the original image or the edited image.
Response: JSON
Example-Response:
    {
        "status": "Success"
    }
<hr>
5) Local Edit:
Description: Allows you to locally edit your image based on parametric values given.
NOTE: You cannot do any globalEdits after calling localEdit.
Url: /localEdit
Type: POST
Body-Type: JSON 
Body:
    {
        "id":"fec42e0311f7e1f0",
        "alpha":[3,2,3],
        "target_":["Hairstyle","Hair Color","Eye Color"],
        "style_idx":[3,2,0],
        "is_styleclip":0,
        "on_original":1
    }

Reference:
    Ethenicity: Australian,Indian,Asian,Latino,Hawaiian,Blonde
    Hair Color: Black Hair Color, Brown Hair Color, Blonde Hair Color, Red Hair Color, White Hair Color, Silver Hair Color
    Hair Highlights: Black Hair Highlights, brown Hair Highlights, Blonde Hair Highlights, Red Hair Highlights, White Hair Highlights, Silver Hair Highlights, golden Hair Color
    Hairstyle: Bald, Bangs, Curly Hair, Hi-top Fade, Fringe Hair, Bob Cut, Mohawk
    Facial Hair: Moustache, Beard
    Accesories: Earrings
    Emotions: Surprised Face, Angry Face, Contemptuous Face, Sad Face, Happy Face, Disgusted Face
    Eye Color: Blue Eyes
    Eye Size: Big Eyes
    Face Shape: Fat Face, Chubby Face
    Makeup: Eye Liner, Eye Makeup, Lips Makeup, Eyebrow Makeup
    Ear Size: Large Ears
    Lip Size: Large Lips
    Facespots: Freckles

Parameters-Description:
    a) id: Session ID generated via /login
    b) alpha: Takes an sequential array of alphas value, The alpha value ranges from -3 to 3, implies how strongly do you want to apply that edit.
    c) target_: Takes an sequential array of target value, The facial attribute you want to change, look up to the Reference section for that. for example: ["Hairstyle", "Eye Color"].
    d) style_idx: Takes an sequential array of style index value, Index of what style do you want to apply on the selected target for the face, look up to the Reference section for the index. for example if you want to apply a 'bob cut' with blue eye color then the "target_" will be ["Hairstyle", "Eye Color"] and "style_idx" will be [5,0].
    e) is_styleclip: Takes a boolean value i.e 0 or 1, True only when you're reusing this API call, else false. for example if you alreadly have edited the hairstyle and now you wish to edit the eye color then is_styleclip will be true in the next call.
    f) on_original: Takes a boolean value i.e 0 or 1, to decide weather to do the operation on the original image or the edited image.
Response: JSON
Example-Response:
    {
        "status": "Success"
    }
<hr>
6) Get Image
Description: Returns the edited Image, you can call this after you make other api calls and get a success response to get the edited image.
Url: /getImage
Example-Url: /getImage
Type: POST
Body-Type: JSON
Body:
    {
        "id":"fec42e0311f7e1f0",
        "get_original": 0
    }

Parameters-Description:
    a) id: Session ID generated via /login
    b) get_original: Takes a boolean value, to send the original image or edited image.
Response: Image
<hr>
7) Remove Background:
Description: Removes the backgroud from the given image.
Url: /removeBackground
Type: POST
Body-Type: JSON 
Body:
    {
        "id":"fec42e0311f7e1f0",
        "image_url":"https://546458-1766807-raikfcquaxqncofqfm.stackpathdns.com/pub/media/wordpress/d5f1425700d7460bb2aa1e1e8e1b7e49.jpg"
    }

Parameters-Description:
    a) id: Session ID generated via /login
    b) image_url: Valid Image url.
Response: JSON
Example-Response:
    {
        "status": "Success"
    }
<hr>
7) Swap Face:
Description: Swap faces of given images.
Url: /faceSwap
Type: POST
Body-Type: JSON 
Body:
    {
        "id":"fec42e0311f7e1f0",
        "src_image_url":"https://546458-1766807-raikfcquaxqncofqfm.stackpathdns.com/pub/media/wordpress/d5f1425700d7460bb2aa1e1e8e1b7e49.jpg"
        "dst_image_url":"https://media.allure.com/photos/57890b5ce1c20f8b68d406a8/master/w_1600%2Cc_limit/celebrity-trends-2016-07-taylor-swift-face-reading.jpg"
    }

Parameters-Description:
    a) id: Session ID generated via /login
    b) src_image_url: Valid Image url of the source image whose face you want to apply.
    c) dst_image_url: Valid Image url of the destination image whose face you wan to replace.
Response: JSON
Example-Response:
    {
        "status": "Success"
    }